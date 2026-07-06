import os
from typing import List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from src.msmarco_utils import load_docs_and_queries_by_split


def get_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, and CPU fallback is disabled. "
            "Please run on a CUDA-ready machine or install a PyTorch build "
            "compatible with the installed NVIDIA driver."
        )
    return "cuda"


DEVICE = get_device()


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), seq_lens
        ]


def encode(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 128,
    instruction=None,
) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        if instruction:
            batch = [instruction + t for t in batch]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        embs = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.paths.data_dir, exist_ok=True)

    split_files = [
        ("train", cfg.paths.msmarco_train),
        ("valid", cfg.paths.msmarco_valid),
        ("test", cfg.paths.msmarco_test),
    ]
    print("Loading MSMARCO data (train + valid + test docs)...")
    docs, queries_by_split, format_by_split = load_docs_and_queries_by_split(split_files)
    for split_name, _ in split_files:
        print(
            f"  {split_name}: format={format_by_split[split_name]}, "
            f"queries={len(queries_by_split[split_name])}"
        )
    print(f"Merged docs: {len(docs)}")
    if not docs:
        raise ValueError(
            "No documents found in MSMARCO splits. Check dataset file paths and format."
        )

    doc_texts = [d["text"] for d in docs]  # use full passage text

    print(f"Loading {cfg.emb.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.emb.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.emb.model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    if os.path.exists(cfg.paths.doc_embeddings):
        print("Loading cached doc embeddings...")
        doc_embs = np.load(cfg.paths.doc_embeddings)
        if doc_embs.shape[0] != len(doc_texts):
            print(
                "Cached embeddings size mismatch "
                f"(cache={doc_embs.shape[0]}, docs={len(doc_texts)}), regenerating..."
            )
            doc_embs = encode(doc_texts, tokenizer, model, batch_size=cfg.emb.batch_size)
            np.save(cfg.paths.doc_embeddings, doc_embs)
            print(
                f"Saved embeddings to {cfg.paths.doc_embeddings}, shape: {doc_embs.shape}"
            )
    else:
        print("Encoding documents...")
        doc_embs = encode(doc_texts, tokenizer, model, batch_size=cfg.emb.batch_size)
        np.save(cfg.paths.doc_embeddings, doc_embs)
        print(f"Saved embeddings to {cfg.paths.doc_embeddings}, shape: {doc_embs.shape}")


if __name__ == "__main__":
    main()
