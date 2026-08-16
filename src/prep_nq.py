"""Prepare NQ320k for the RQ / analysis pipeline.

Chosen from the representation probe (src/nq_doc_repr_probe.py):
  encoder  = GTR-T5-large   (gold R@1 0.75 vs Qwen 0.46 on a 20k subset)
  document = first 2000 chars -- reading more HURT, because a Wikipedia lead
             section already summarises the page while the tail is tables and
             references.

Writes, all sharing one document index order:
  docs.jsonl, doc_embeddings.npy, {train,test}_queries.jsonl (with gt_doc_idx),
  {train,test}_query_embeddings.npy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/nq320k/raw")
    ap.add_argument("--out-dir", default="data/nq320k")
    ap.add_argument("--model", default="sentence-transformers/gtr-t5-large")
    ap.add_argument("--doc-chars", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def write_oracle_ceiling(out_dir, result):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "oracle_ceiling.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


@torch.no_grad()
def encode(texts, tok, model, args, max_len, desc):
    out = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc=desc):
        enc = tok(texts[i : i + args.batch_size], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(args.device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(F.normalize(e, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    corpus = [json.loads(l) for l in open(f"{args.raw_dir}/corpus.jsonl")]
    doc_idx = {d["docid"]: i for i, d in enumerate(corpus)}
    print(f"docs: {len(corpus):,}")

    with open(f"{args.out_dir}/docs.jsonl", "w") as f:
        for d in corpus:
            f.write(json.dumps({"docid": d["docid"],
                                "text": d["document"][: args.doc_chars]}) + "\n")

    splits = {}
    for raw, name in (("train", "train"), ("valid", "test")):
        rows = [json.loads(l) for l in open(f"{args.raw_dir}/{raw}.jsonl")]
        rows = [r for r in rows if r["docid"] in doc_idx]
        with open(f"{args.out_dir}/{name}_queries.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"query": r["query"], "docid": r["docid"],
                                    "gt_doc_idx": doc_idx[r["docid"]]}) + "\n")
        splits[name] = rows
        print(f"{name} queries: {len(rows):,}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
    if hasattr(model, "encoder"):
        model = model.encoder
    model = model.to(args.device).eval()

    D = encode([d["document"][: args.doc_chars] for d in corpus],
               tok, model, args, 512, "docs")
    np.save(f"{args.out_dir}/doc_embeddings.npy", D)
    print(f"saved doc_embeddings {D.shape}")

    for name in ("train", "test"):
        Q = encode([r["query"] for r in splits[name]], tok, model, args, 128,
                   f"{name} queries")
        np.save(f"{args.out_dir}/{name}_query_embeddings.npy", Q)
        np.save(f"{args.out_dir}/{name}_query_docidx.npy",
                np.array([doc_idx[r["docid"]] for r in splits[name]], dtype=np.int64))
        print(f"saved {name}_query_embeddings {Q.shape}")

    # true oracle ceiling on the full corpus -- the 20k probe was optimistic
    dev = torch.from_numpy(
        np.load(f"{args.out_dir}/test_query_embeddings.npy")
    ).to(args.device)
    Dg = torch.from_numpy(D).to(args.device)
    gold = torch.from_numpy(
        np.load(f"{args.out_dir}/test_query_docidx.npy")
    ).to(args.device)
    hits = {k: 0 for k in (1, 5, 10, 100)}
    for i in range(0, len(dev), 256):
        s = dev[i : i + 256] @ Dg.t()
        gi = gold[i : i + 256]
        rank = (s > s.gather(1, gi[:, None])).sum(1) + 1
        for k in hits:
            hits[k] += (rank <= k).sum().item()
    res = {f"gold_recall@{k}": v / len(dev) for k, v in hits.items()}
    write_oracle_ceiling(args.out_dir, res)
    print("\nfull-corpus oracle ceiling:")
    for k, v in res.items():
        print(f"  {k:18s} {v:.4f}")


if __name__ == "__main__":
    main()
