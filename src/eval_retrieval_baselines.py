"""Evaluate CPU BM25 and pretrained DPR on frozen GR corpora and test queries."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def strip_query_prefix(text):
    text = str(text).strip()
    return text[6:].strip() if text.lower().startswith("query:") else text


def load_corpus(path):
    documents, docids = [], []
    with open(path) as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            text = row.get("text", row.get("document"))
            if text is None:
                raise ValueError(f"corpus row {idx} has neither text nor document")
            documents.append(str(text))
            docids.append(str(row.get("docid", idx)))
    if not documents or len(set(docids)) != len(docids):
        raise ValueError("corpus must be nonempty with unique document IDs")
    return documents, docids


def load_queries(path, n_documents):
    queries, golds = [], []
    with open(path) as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            query = row.get("input", row.get("query"))
            if query is None:
                raise ValueError(f"query row {idx} has no input/query field")
            if "gt_doc_indices" in row:
                gold = {int(value) for value in row["gt_doc_indices"]}
            elif "gt_doc_idx" in row:
                gold = {int(row["gt_doc_idx"])}
            else:
                raise ValueError(f"query row {idx} has no document-index gold label")
            if not gold or any(value < 0 or value >= n_documents for value in gold):
                raise ValueError(f"query row {idx} has invalid gold indices {gold}")
            queries.append(strip_query_prefix(query))
            golds.append(gold)
    if not queries:
        raise ValueError("query file is empty")
    return queries, golds


def ranking_metrics(rankings, golds):
    if len(rankings) != len(golds) or not rankings:
        raise ValueError("rankings and golds must have equal nonzero length")
    totals = {"R@1": 0.0, "R@10": 0.0, "R@100": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0}
    for ranking, gold in zip(rankings, golds):
        for cutoff in (1, 10, 100):
            totals[f"R@{cutoff}"] += len(set(ranking[:cutoff]) & gold) / len(gold)
        relevant_ranks = [rank for rank, doc_idx in enumerate(ranking[:10], 1) if doc_idx in gold]
        if relevant_ranks:
            totals["MRR@10"] += 1.0 / relevant_ranks[0]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
        totals["nDCG@10"] += dcg / ideal
    return {key: value / len(rankings) for key, value in totals.items()}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_bm25(documents, queries, top_k):
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    started = time.perf_counter()
    corpus_tokens = bm25s.tokenize(documents, stopwords="en", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens, show_progress=True)
    index_seconds = time.perf_counter() - started
    query_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=True)
    started = time.perf_counter()
    results, _ = retriever.retrieve(query_tokens, k=top_k, show_progress=True)
    search_seconds = time.perf_counter() - started
    return np.asarray(results, dtype=np.int64), {
        "implementation": "bm25s",
        "bm25_method": "lucene",
        "tokenization": "English stopwords and PyStemmer English stemming",
        "index_seconds": index_seconds,
        "search_seconds": search_seconds,
    }


def _batches(items, size):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def encode_dpr_corpus(
    documents, model_name, cache_path, batch_size, max_length, device, corpus_sha256
):
    import torch
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

    metadata_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    expected = {
        "model": model_name,
        "n_documents": len(documents),
        "max_length": max_length,
        "corpus_sha256": corpus_sha256,
    }
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in expected.items()):
            return np.load(cache_path, mmap_mode="r"), True
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name).to(device).eval()
    output = None
    with torch.inference_mode():
        for start, batch in _batches(documents, batch_size):
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            vectors = model(**tokens).pooler_output.float().cpu().numpy()
            if output is None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                output = np.lib.format.open_memmap(cache_path, mode="w+", dtype="float32", shape=(len(documents), vectors.shape[1]))
            output[start : start + len(batch)] = vectors
            if start % (batch_size * 100) == 0:
                print(f"DPR corpus encoded {start + len(batch):,}/{len(documents):,}", flush=True)
    output.flush()
    metadata_path.write_text(json.dumps({**expected, "dimension": int(output.shape[1])}, indent=2))
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.load(cache_path, mmap_mode="r"), False


def encode_dpr_queries(queries, model_name, batch_size, max_length, device):
    import torch
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

    tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_name)
    model = DPRQuestionEncoder.from_pretrained(model_name).to(device).eval()
    outputs = []
    with torch.inference_mode():
        for _, batch in _batches(queries, batch_size):
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs.append(model(**tokens).pooler_output.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_dpr(documents, queries, args, corpus_sha256):
    import faiss

    started = time.perf_counter()
    vectors, reused = encode_dpr_corpus(
        documents, args.ctx_model, args.embedding_cache, args.batch_size,
        args.context_max_length, args.device, corpus_sha256,
    )
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    index = faiss.IndexFlatIP(vectors.shape[1])
    for start in range(0, len(vectors), 50000):
        index.add(np.asarray(vectors[start : start + 50000], dtype="float32"))
    query_vectors = encode_dpr_queries(
        queries, args.question_model, args.batch_size,
        args.query_max_length, args.device,
    )
    _, results = index.search(np.asarray(query_vectors, dtype="float32"), args.top_k)
    search_seconds = time.perf_counter() - started
    return results, {
        "implementation": "transformers DPR + exact FAISS IndexFlatIP",
        "ctx_model": args.ctx_model,
        "question_model": args.question_model,
        "task_specific_fine_tuning": False,
        "checkpoint_interpretation": "pretrained DPR multiset transfer baseline",
        "context_max_length": args.context_max_length,
        "query_max_length": args.query_max_length,
        "embedding_cache": str(args.embedding_cache),
        "embedding_cache_reused": reused,
        "corpus_encode_seconds": encode_seconds,
        "index_and_search_seconds": search_seconds,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bm25", "dpr"), required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-max-length", type=int, default=256)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--ctx-model", default="facebook/dpr-ctx_encoder-multiset-base")
    parser.add_argument("--question-model", default="facebook/dpr-question_encoder-multiset-base")
    parser.add_argument("--embedding-cache", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k < 100:
        raise ValueError("top-k must be at least 100 for the requested metrics")
    if args.method == "dpr" and args.embedding_cache is None:
        raise ValueError("DPR requires --embedding-cache")
    documents, docids = load_corpus(args.corpus)
    queries, golds = load_queries(args.queries, len(documents))
    corpus_sha256 = file_sha256(args.corpus)
    queries_sha256 = file_sha256(args.queries)
    print(f"Loaded {len(documents):,} documents and {len(queries):,} queries")
    if args.method == "bm25":
        rankings, details = evaluate_bm25(documents, queries, args.top_k)
    else:
        rankings, details = evaluate_dpr(documents, queries, args, corpus_sha256)
    metrics = ranking_metrics(rankings.tolist(), golds)
    result = {
        "dataset": args.dataset_name,
        "method": args.method,
        "n_documents": len(documents),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "corpus_path": str(args.corpus),
        "queries_path": str(args.queries),
        "corpus_sha256": corpus_sha256,
        "queries_sha256": queries_sha256,
        "gold_semantics": "document index; one or more gold documents per query",
        "metrics": metrics,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
