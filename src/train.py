import glob
import inspect
import json
import logging
import os
import re
from typing import Dict, List, Optional

# Required by accelerate/NCCL on RTX 4000 series GPUs.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import hydra
import torch
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LogitsProcessorList,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

from .dataset import GRDataset
from .multi_view_tokenizer import (
    add_multi_view_special_tokens,
    build_multi_view_special_tokens,
    validate_atomic_tokens,
)

log = logging.getLogger(__name__)


def _report_to_wandb(report_to) -> bool:
    if isinstance(report_to, str):
        return report_to.lower() == "wandb"
    return "wandb" in report_to


def route_view_index(route: str, n_views: int, n_levels: int) -> int:
    if n_views <= 0 or n_levels <= 0 or n_levels % n_views != 0:
        raise ValueError(
            f"invalid view/layer shape: n_views={n_views}, n_levels={n_levels}"
        )
    layers = re.findall(r"<r(\d+)_\d+>", route)
    if not layers:
        raise ValueError(f"route contains no RQ layer tokens: {route!r}")
    first_layer = int(layers[0])
    levels_per_view = n_levels // n_views
    view = first_layer // levels_per_view
    expected_layers = list(
        range(view * levels_per_view, (view + 1) * levels_per_view)
    )
    if view >= n_views or [int(layer) for layer in layers] != expected_layers:
        raise ValueError(f"route violates view/layer namespace contract: {route!r}")
    return view


class ConstrainedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        gt_rqids: List[str],
        eval_rqids_by_view: List[List[str]],
        eval_n_levels: int,
        eval_tokenizer: AutoTokenizer,
        eval_beams: int = 10,
        max_out_len: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.processed_input_tokens = 0
        self.processed_target_tokens = 0
        self.gt_rqids = gt_rqids
        self.eval_rqids_by_view = eval_rqids_by_view
        self.eval_n_levels = eval_n_levels
        self._fast_procs = None
        self.eval_tokenizer = eval_tokenizer
        self.eval_beams = eval_beams
        self.max_out_len = max_out_len

    def compute_loss(self, model, inputs, *args, **kwargs):
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        if attention_mask is not None:
            self.processed_input_tokens += int(attention_mask.sum().item())
        if labels is not None:
            self.processed_target_tokens += int((labels != -100).sum().item())
        return super().compute_loss(model, inputs, *args, **kwargs)

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[GRDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        del ignore_keys, kwargs  # compatibility with newer Trainer.evaluate() kwargs
        if self.args.world_size != 1:
            raise RuntimeError(
                "custom constrained evaluation currently supports world_size=1 only"
            )
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()
        if self._fast_procs is None:
            from .fast_trie import FastRQTrie, FastRQTrieLogitsProcessor

            self._fast_procs = [
                FastRQTrieLogitsProcessor(
                    FastRQTrie(
                        routes,
                        self.eval_tokenizer,
                        self.eval_tokenizer.eos_token_id,
                    ),
                    len(self.eval_tokenizer),
                    self.model.device,
                )
                for routes in self.eval_rqids_by_view
            ]

        hits: Dict[int, float] = {1: 0, 5: 0, 10: 0}
        n: int = 0

        for batch in dataloader:
            batch = self._prepare_inputs(batch)
            bs = batch["input_ids"].shape[0]
            row_views = [
                route_view_index(
                    self.gt_rqids[n + row],
                    len(self._fast_procs),
                    self.eval_n_levels,
                )
                for row in range(bs)
            ]
            beams_by_row = [None] * bs

            for view, processor in enumerate(self._fast_procs):
                row_indices = [
                    row for row, row_view in enumerate(row_views) if row_view == view
                ]
                if not row_indices:
                    continue
                tensor_indices = torch.tensor(
                    row_indices, dtype=torch.long, device=batch["input_ids"].device
                )
                out = self.model.generate(
                    input_ids=batch["input_ids"].index_select(0, tensor_indices),
                    attention_mask=batch["attention_mask"].index_select(
                        0, tensor_indices
                    ),
                    num_beams=self.eval_beams,
                    num_return_sequences=self.eval_beams,
                    max_new_tokens=self.max_out_len,
                    logits_processor=LogitsProcessorList([processor]),
                )
                decoded = self.eval_tokenizer.batch_decode(
                    out, skip_special_tokens=False
                )
                for local_row, original_row in enumerate(row_indices):
                    beams_by_row[original_row] = [
                        " ".join(
                            re.findall(
                                r"<r\d+_\d+>",
                                decoded[
                                    local_row * self.eval_beams + beam_index
                                ],
                            )
                        )
                        for beam_index in range(self.eval_beams)
                    ]

            for row in range(bs):
                gt = self.gt_rqids[n + row]
                beams = beams_by_row[row]
                if beams is None:
                    raise RuntimeError(f"no view-specific decode for eval row {n + row}")
                for k in [1, 5, 10]:
                    if gt in beams[:k]:
                        hits[k] += 1

            n += bs

        self.model.train()

        metrics = {
            f"{metric_key_prefix}_hits@{k}": round(hits[k] / n, 4) for k in [1, 5, 10]
        }
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics


def validate_multi_view_training_contract(enabled, n_views, n_levels):
    if not enabled:
        return
    if n_views != 3:
        raise ValueError(
            f"Multi-DocID training requires exactly 3 views; got {n_views}"
        )
    if n_levels % n_views != 0:
        raise ValueError(
            f"n_levels={n_levels} must be divisible by n_views={n_views}"
        )


def find_best_checkpoint(out_dir: str):
    """Return best_model_checkpoint from the numerically latest trainer state."""
    states = glob.glob(f"{out_dir}/checkpoint-*/trainer_state.json")
    if not states:
        return None

    def checkpoint_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else -1

    latest_state = max(states, key=checkpoint_step)
    with open(latest_state) as f:
        return json.load(f).get("best_model_checkpoint")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    multi_view_cfg = cfg.get("multi_view", {})
    multi_view_enabled = bool(multi_view_cfg.get("enabled", False))
    n_views = int(multi_view_cfg.get("n_views", 3)) if multi_view_enabled else 1
    validate_multi_view_training_contract(
        multi_view_enabled, n_views, int(cfg.rq.n_levels)
    )
    rq_tokens = [
        f"<r{level}_{c}>"
        for level in range(cfg.rq.n_levels)
        for c in range(cfg.rq.n_codes)
    ]
    special_tokens = (
        build_multi_view_special_tokens(
            cfg.rq.n_levels, cfg.rq.n_codes, n_views
        )
        if multi_view_enabled
        else rq_tokens
    )

    best_ckpt = find_best_checkpoint(cfg.out_dir)
    model_path = best_ckpt or cfg.base_model
    log.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if best_ckpt is None:
        n_added = add_multi_view_special_tokens(tokenizer, special_tokens)
        log.info(f"  Added {n_added} special tokens")
    validate_atomic_tokens(tokenizer, special_tokens)

    log.info(f"  Vocab size: {len(tokenizer)}")

    log.info(f"Loading model from {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    base_vocab_size = model.get_input_embeddings().num_embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Seed the new DocID token embeddings with their RQ codebook vectors.
    # T5-large's d_model equals the codebook dimension and the embeddings are
    # tied, so the decoder logit for code k becomes <hidden, centroid_k>:
    # nearest-centroid decoding is available from step 0, and semantically close
    # codes start close together instead of at random points.
    if best_ckpt is None and cfg.get("codebook_init", None):
        import numpy as np

        cb = np.load(cfg.codebook_init)  # (n_levels, n_codes, D)
        emb = model.get_input_embeddings().weight.data
        base_rows = base_vocab_size
        target_norm = emb[:base_rows].norm(dim=-1).mean()
        n_set = 0
        for level in range(cb.shape[0]):
            vecs = torch.from_numpy(cb[level]).float()
            vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            vecs = vecs * target_norm * float(cfg.get("codebook_init_scale", 1.0))
            for code in range(cb.shape[1]):
                tid = tokenizer.convert_tokens_to_ids(f"<r{level}_{code}>")
                if tid is not None and 0 <= tid < emb.shape[0]:
                    emb[tid] = vecs[code].to(emb.dtype)
                    n_set += 1
        log.info(f"  codebook-init: seeded {n_set} DocID tokens, target norm "
                 f"{target_norm:.1f} x {float(cfg.get('codebook_init_scale', 1.0))}")

    log.info("Building RQ trie...")

    with open(cfg.paths.idx_to_rqid) as f:
        idx_to_rqid = json.load(f)
    eval_rqids_by_view = [[] for _ in range(n_views)]
    for route in idx_to_rqid:
        view = route_view_index(route, n_views, int(cfg.rq.n_levels))
        eval_rqids_by_view[view].append(route)
    if any(not routes for routes in eval_rqids_by_view):
        raise ValueError("every configured view must have at least one evaluation route")
    log.info(
        "  Loaded evaluation DocIDs by view: "
        + ", ".join(str(len(set(routes))) for routes in eval_rqids_by_view)
    )

    log.info("Loading datasets...")
    train_ds = GRDataset(
        cfg.paths.train_data, tokenizer, cfg.data.max_in_len, cfg.data.max_out_len
    )
    test_ds = GRDataset(
        cfg.paths.test_data, tokenizer, cfg.data.max_in_len, cfg.data.max_out_len
    )
    with open(cfg.paths.test_data) as f:
        gt_rqids = [json.loads(line)["gt_rqid"] for line in f]

    log.info(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    t = cfg.training
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.out_dir,
        num_train_epochs=t.epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,  # small: generates eval_beams seqs per input
        gradient_accumulation_steps=t.grad_accum,
        learning_rate=t.lr,
        warmup_ratio=t.warmup,
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        bf16=True,
        predict_with_generate=False,  # generation handled in evaluate()
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_hits@10",
        greater_is_better=True,
        logging_steps=t.logging_steps,
        report_to=t.report_to,
        run_name=cfg.wandb.run_name,
        dataloader_num_workers=4,
    )

    if _report_to_wandb(t.report_to):
        import wandb

        wandb_dir = os.path.abspath(os.path.join(cfg.out_dir, "wandb"))
        wandb_cache_dir = os.path.join(wandb_dir, "cache")
        wandb_config_dir = os.path.join(wandb_dir, "config")
        wandb_data_dir = os.path.join(wandb_dir, "data")
        for path in [wandb_dir, wandb_cache_dir, wandb_config_dir, wandb_data_dir]:
            os.makedirs(path, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
        os.environ["WANDB_CONFIG_DIR"] = wandb_config_dir
        os.environ["WANDB_DATA_DIR"] = wandb_data_dir
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            dir=wandb_dir,
            settings=wandb.Settings(root_dir=wandb_dir, x_files_dir=wandb_dir),
        )

    # transformers>=5 renamed Trainer's tokenizer arg to processing_class.
    trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_processing_kwargs = (
        {"processing_class": tokenizer}
        if "processing_class" in trainer_init_params
        else {"tokenizer": tokenizer}
    )

    trainer = ConstrainedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        **trainer_processing_kwargs,
        data_collator=collator,
        gt_rqids=gt_rqids,
        eval_rqids_by_view=eval_rqids_by_view,
        eval_n_levels=int(cfg.rq.n_levels),
        eval_tokenizer=tokenizer,
        eval_beams=t.eval_beams,
        max_out_len=cfg.data.max_out_len,
    )

    log.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model(cfg.out_dir)
    token_ids_before_save = {
        token: tokenizer.convert_tokens_to_ids(token) for token in special_tokens
    }
    tokenizer.save_pretrained(cfg.out_dir)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(cfg.out_dir)
    validate_atomic_tokens(reloaded_tokenizer, special_tokens)
    token_ids_after_reload = {
        token: reloaded_tokenizer.convert_tokens_to_ids(token)
        for token in special_tokens
    }
    if token_ids_after_reload != token_ids_before_save:
        raise RuntimeError("special-token IDs changed after tokenizer save/reload")
    training_manifest = {
        "shared_model_instances": 1,
        "multi_view_enabled": multi_view_enabled,
        "n_views": n_views,
        "n_train_examples": len(train_ds),
        "n_eval_examples": len(test_ds),
        "configured_epochs": float(t.epochs),
        "completed_epochs": float(trainer.state.epoch or 0.0),
        "optimizer_steps": int(trainer.state.global_step),
        "world_size": int(training_args.world_size),
        "processed_input_tokens_local": trainer.processed_input_tokens,
        "processed_target_tokens_local": trainer.processed_target_tokens,
        "processed_tokens_local": (
            trainer.processed_input_tokens + trainer.processed_target_tokens
        ),
        "train_metrics": train_result.metrics,
    }
    with open(os.path.join(cfg.out_dir, "training_manifest.json"), "w") as handle:
        json.dump(training_manifest, handle, indent=2)
    log.info(f"Model saved → {cfg.out_dir}")


if __name__ == "__main__":
    main()
