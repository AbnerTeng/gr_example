import os
import json
from typing import List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    print("Loading MSMARCO data...")
    with open(cfg.paths.msmarco_train) as f:
        raw = [json.loads(line) for line in f]

    docs = {d["doc_id"]: d for d in raw if d.get("operation") == "indexing"}
    queries = [d for d in raw if d.get("operation") == "query"]
    print(f"Docs: {len(docs)}, Queries: {len(queries)}")

    doc_list = list(docs.values())
    doc_texts = [d["text"] for d in doc_list]  # use full passage text

    print(f"Loading {cfg.emb.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.emb.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.emb.model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    if os.path.exists(cfg.paths.doc_embeddings):
        print("Loading cached doc embeddings...")
        doc_embs = np.load(cfg.paths.doc_embeddings)
    else:
        print("Encoding documents...")
        doc_embs = encode(doc_texts, tokenizer, model, batch_size=cfg.emb.batch_size)
        np.save(cfg.paths.doc_embeddings, doc_embs)
        print(f"Saved embeddings to {cfg.paths.doc_embeddings}, shape: {doc_embs.shape}")


if __name__ == "__main__":
    main()
