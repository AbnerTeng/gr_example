"""Embed the fixed-order MSMARCO-300K corpus and its train/dev queries."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def stable_docid_hash(docids):
    digest = hashlib.sha256()
    for docid in docids:
        digest.update(str(docid).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_corpus(path):
    docids = []
    texts = []
    doc_to_idx = {}
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            docid = str(row["docid"])
            text = row["document"]
            if docid in doc_to_idx:
                raise ValueError(f"duplicate docid {docid!r} at line {line_number}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"empty document text for docid {docid!r}")
            doc_to_idx[docid] = len(docids)
            docids.append(docid)
            texts.append(text)
    return docids, texts, doc_to_idx


def load_query_pairs(path, doc_to_idx):
    texts = []
    docidx = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            docid = str(row["docid"])
            query = row["query"]
            if docid not in doc_to_idx:
                raise ValueError(
                    f"query target {docid!r} at line {line_number} is absent from corpus"
                )
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"empty query at line {line_number}")
            texts.append(query)
            docidx.append(doc_to_idx[docid])
    return texts, np.asarray(docidx, dtype=np.int64)


def encode_to_npy(texts, output, tokenizer, model, device, batch_size, max_length):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    if not texts:
        raise ValueError("cannot embed an empty text collection")
    hidden_size = getattr(model.config, "d_model", None) or model.config.hidden_size
    output = Path(output)
    partial = output.with_suffix(output.suffix + ".partial")
    array = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(texts), hidden_size),
    )
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=output.stem):
            end = min(start + batch_size, len(texts))
            encoded = tokenizer(
                texts[start:end],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            array[start:end] = F.normalize(embeddings, dim=-1).float().cpu().numpy()
    array.flush()
    del array
    os.replace(partial, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/msmarco300k"))
    parser.add_argument("--model", default="sentence-transformers/gtr-t5-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--doc-batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--doc-max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    args = parse_args()
    paths = {
        "doc_embeddings": args.data_dir / "doc_embeddings.npy",
        "train_query_embeddings": args.data_dir / "train_query_embeddings.npy",
        "train_query_docidx": args.data_dir / "train_query_docidx.npy",
        "dev_query_embeddings": args.data_dir / "dev_query_embeddings.npy",
        "dev_query_docidx": args.data_dir / "dev_query_docidx.npy",
        "docids": args.data_dir / "docids.json",
        "metadata": args.data_dir / "embedding_metadata.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to mix old and new embedding artifacts; existing: "
            + ", ".join(existing)
        )

    docids, documents, doc_to_idx = load_corpus(args.data_dir / "corpus.jsonl")
    train_queries, train_docidx = load_query_pairs(
        args.data_dir / "train.jsonl", doc_to_idx
    )
    dev_queries, dev_docidx = load_query_pairs(
        args.data_dir / "dev.jsonl", doc_to_idx
    )
    print(
        f"documents={len(documents):,} train_queries={len(train_queries):,} "
        f"dev_queries={len(dev_queries):,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
    if hasattr(model, "encoder"):
        model = model.encoder
    model = model.to(args.device).eval()

    encode_to_npy(
        documents,
        paths["doc_embeddings"],
        tokenizer,
        model,
        args.device,
        args.doc_batch_size,
        args.doc_max_length,
    )
    encode_to_npy(
        train_queries,
        paths["train_query_embeddings"],
        tokenizer,
        model,
        args.device,
        args.query_batch_size,
        args.query_max_length,
    )
    encode_to_npy(
        dev_queries,
        paths["dev_query_embeddings"],
        tokenizer,
        model,
        args.device,
        args.query_batch_size,
        args.query_max_length,
    )
    np.save(paths["train_query_docidx"], train_docidx)
    np.save(paths["dev_query_docidx"], dev_docidx)
    with open(paths["docids"], "w") as handle:
        json.dump(docids, handle)
    metadata = {
        "model": args.model,
        "n_documents": len(documents),
        "n_train_queries": len(train_queries),
        "n_dev_queries": len(dev_queries),
        "docid_order_sha256": stable_docid_hash(docids),
        "embedding_dimension": int(np.load(paths["doc_embeddings"], mmap_mode="r").shape[1]),
        "doc_max_length": args.doc_max_length,
        "query_max_length": args.query_max_length,
    }
    with open(paths["metadata"], "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
