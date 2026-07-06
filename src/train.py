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
from .inference import build_rq_trie, RQTrieLogitsProcessor, TrieNode

log = logging.getLogger(__name__)


def _report_to_wandb(report_to) -> bool:
    if isinstance(report_to, str):
        return report_to.lower() == "wandb"
    return "wandb" in report_to


class ConstrainedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        trie_root: TrieNode,
        gt_rqids: List[str],
        eval_tokenizer: AutoTokenizer,
        eval_beams: int = 10,
        max_out_len: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trie_root = trie_root
        self.gt_rqids = gt_rqids
        self.eval_tokenizer = eval_tokenizer
        self.eval_beams = eval_beams
        self.max_out_len = max_out_len

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[GRDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        del ignore_keys, kwargs  # compatibility with newer Trainer.evaluate() kwargs
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()
        logits_proc = LogitsProcessorList(
            [RQTrieLogitsProcessor(self.trie_root, self.eval_tokenizer.eos_token_id)]
        )

        hits: Dict[int, float] = {1: 0, 5: 0, 10: 0}
        n: int = 0

        for batch in dataloader:
            batch = self._prepare_inputs(batch)
            bs = batch["input_ids"].shape[0]
            out = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=self.eval_beams,
                num_return_sequences=self.eval_beams,
                max_new_tokens=self.max_out_len,
                logits_processor=logits_proc,
            )
            decoded = self.eval_tokenizer.batch_decode(out, skip_special_tokens=False)

            for i in range(bs):
                gt = self.gt_rqids[n + i]
                beams = [
                    " ".join(
                        re.findall(r"<r\d+_\d+>", decoded[i * self.eval_beams + j])
                    )
                    for j in range(self.eval_beams)
                ]
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


def find_best_checkpoint(out_dir: str):
    """Return best_model_checkpoint from the most recent trainer_state.json, or None."""
    states = sorted(glob.glob(f"{out_dir}/checkpoint-*/trainer_state.json"))

    if not states:
        return None
    with open(states[-1]) as f:
        return json.load(f).get("best_model_checkpoint")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    rq_tokens = [
        f"<r{level}_{c}>"
        for level in range(cfg.rq.n_levels)
        for c in range(cfg.rq.n_codes)
    ]

    best_ckpt = find_best_checkpoint(cfg.out_dir)
    model_path = best_ckpt or cfg.base_model
    log.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if best_ckpt is None:
        tokenizer.add_tokens(rq_tokens, special_tokens=True)

    log.info(f"  Vocab size: {len(tokenizer)}")

    log.info(f"Loading model from {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    log.info("Building RQ trie...")

    with open(cfg.paths.idx_to_rqid) as f:
        idx_to_rqid = json.load(f)

    trie_root = build_rq_trie(idx_to_rqid, tokenizer)
    log.info(f"  Trie built from {len(set(idx_to_rqid))} unique docids")

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
        trie_root=trie_root,
        gt_rqids=gt_rqids,
        eval_tokenizer=tokenizer,
        eval_beams=t.eval_beams,
        max_out_len=cfg.data.max_out_len,
    )

    log.info("Starting training...")
    trainer.train()
    trainer.save_model(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    log.info(f"Model saved → {cfg.out_dir}")


if __name__ == "__main__":
    main()
