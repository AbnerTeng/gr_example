import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import faiss
except ImportError:
    faiss = None

from src.msmarco_utils import load_docs_and_queries_by_split


RQ_PATTERN = re.compile(r"<r\d+_(\d+)>")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether query embedding top-K retrieved documents share "
            "RQ DocID prefixes with each query's ground-truth document."
        )
    )
    parser.add_argument("--doc-embeddings", type=str, default="data/doc_embeddings.npy")
    parser.add_argument("--query-embeddings", type=str, default="data/test_query_embeddings.npy")
    parser.add_argument("--rq-codes", type=str, default="data/rq_codes.npy")
    parser.add_argument("--idx-to-rqid", type=str, default="data/idx_to_rqid.json")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/query_rq_prefix")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--search-batch-size", type=int, default=8192)
    parser.add_argument(
        "--k-values",
        type=str,
        default="10,50,100",
        help="Comma-separated K values, e.g. 10,50,100.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Maximum prefix levels to evaluate. Uses min(levels, rq_code_width).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=-1,
        help="Use only the first N test queries. -1 means all queries.",
    )
    parser.add_argument(
        "--query-instruction",
        type=str,
        default="",
        help="Optional prefix prepended before each query when encoding.",
    )
    parser.add_argument(
        "--keep-query-prefix",
        action="store_true",
        help='Keep the leading "query: " text in data/test.jsonl inputs.',
    )
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Require --query-embeddings to exist instead of encoding missing queries.",
    )
    parser.add_argument(
        "--recompute-query-embeddings",
        action="store_true",
        help="Force regeneration of query embeddings even if cache exists.",
    )
    parser.add_argument(
        "--exclude-gt-doc",
        action="store_true",
        help="Exclude each query's ground-truth document from its retrieved top-K list.",
    )
    parser.add_argument(
        "--search-extra",
        type=int,
        default=32,
        help="Extra candidates to retrieve when --exclude-gt-doc is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--msmarco-train", type=str, default="data/msmarco/train.jsonl")
    parser.add_argument("--msmarco-valid", type=str, default="data/msmarco/valid.jsonl")
    parser.add_argument("--msmarco-test", type=str, default="data/msmarco/test.jsonl")
    return parser.parse_args()


def parse_k_values(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"K values must be positive. Got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one K value is required.")
    return sorted(set(values))


def parse_rqid(rqid: str) -> List[int]:
    tokens = [int(m.group(1)) for m in RQ_PATTERN.finditer(rqid)]
    if not tokens:
        raise ValueError(f"Cannot parse RQ ID tokens from: {rqid}")
    return tokens


def load_rq_codes(rq_codes_path: Path, idx_to_rqid_path: Path) -> np.ndarray:
    if rq_codes_path.exists():
        rq_codes = np.load(rq_codes_path)
        if rq_codes.ndim != 2:
            raise ValueError(f"Expected 2D rq_codes, got shape={rq_codes.shape}")
        return rq_codes.astype(np.int32)

    if not idx_to_rqid_path.exists():
        raise FileNotFoundError(
            f"Neither {rq_codes_path} nor {idx_to_rqid_path} exists, cannot load RQ tokens."
        )

    with open(idx_to_rqid_path, encoding="utf-8") as f:
        idx_to_rqid = json.load(f)
    if not isinstance(idx_to_rqid, list) or not idx_to_rqid:
        raise ValueError("idx_to_rqid.json should be a non-empty list.")

    parsed = [parse_rqid(rqid) for rqid in idx_to_rqid]
    width = len(parsed[0])
    if any(len(row) != width for row in parsed):
        raise ValueError("Inconsistent RQ token width in idx_to_rqid.json")

    return np.asarray(parsed, dtype=np.int32)


def l2_normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Found zero-norm embedding vector(s).")
    return embs / norms


def load_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_query_input(text: str, keep_query_prefix: bool) -> str:
    if keep_query_prefix:
        return text
    prefix = "query:"
    if text.lower().startswith(prefix):
        return text[len(prefix) :].strip()
    return text.strip()


def load_test_queries(
    path: str, max_queries: int, keep_query_prefix: bool
) -> Tuple[List[str], np.ndarray, List[str]]:
    rows = load_jsonl(path)
    if max_queries > 0:
        rows = rows[:max_queries]
    if not rows:
        raise ValueError(f"No test queries loaded from {path}")

    queries: List[str] = []
    gt_codes: List[List[int]] = []
    gt_semids: List[str] = []

    for row in rows:
        query_text = row.get("input")
        gt_rqid = row.get("gt_rqid") or row.get("output")
        gt_semid = row.get("gt_semid")
        if not isinstance(query_text, str) or not isinstance(gt_rqid, str):
            raise ValueError("Each test row must contain string fields: input and gt_rqid/output.")
        queries.append(normalize_query_input(query_text, keep_query_prefix))
        gt_codes.append(parse_rqid(gt_rqid))
        gt_semids.append(gt_semid if isinstance(gt_semid, str) else "")

    width = len(gt_codes[0])
    if any(len(row) != width for row in gt_codes):
        raise ValueError("Inconsistent GT RQ token width in test data.")

    return queries, np.asarray(gt_codes, dtype=np.int32), gt_semids


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), seq_lens
    ]


def encode_queries(
    queries: Sequence[str],
    model_name: str,
    batch_size: int,
    instruction: str,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("--query-batch-size must be > 0")

    print(f"Loading query embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(DEVICE)
    model.eval()

    all_embs = []
    for start in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
        batch = list(queries[start : start + batch_size])
        if instruction:
            batch = [instruction + q for q in batch]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        embs = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


def load_or_encode_query_embeddings(
    queries: Sequence[str],
    path: Path,
    model_name: str,
    batch_size: int,
    instruction: str,
    no_encode: bool,
    recompute: bool,
) -> np.ndarray:
    if path.exists() and not recompute:
        query_embs = np.load(path).astype(np.float32)
        if query_embs.shape[0] >= len(queries):
            if query_embs.shape[0] > len(queries):
                query_embs = query_embs[: len(queries)]
            print(f"Loaded cached query embeddings: {path}, shape={query_embs.shape}")
            return query_embs
        print(
            f"Query embedding cache size mismatch "
            f"(cache={query_embs.shape[0]}, queries={len(queries)})."
        )
        if no_encode:
            raise ValueError("Cache mismatch and --no-encode was set.")

    if no_encode:
        raise FileNotFoundError(f"Query embeddings not found: {path}")

    query_embs = encode_queries(
        queries=queries,
        model_name=model_name,
        batch_size=batch_size,
        instruction=instruction,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, query_embs)
    print(f"Saved query embeddings: {path}, shape={query_embs.shape}")
    return query_embs


def build_semid_to_doc_index(args: argparse.Namespace) -> Dict[str, int]:
    split_files = [
        ("train", args.msmarco_train),
        ("valid", args.msmarco_valid),
        ("test", args.msmarco_test),
    ]
    docs, _, _ = load_docs_and_queries_by_split(split_files)
    return {str(doc["doc_id"]): i for i, doc in enumerate(docs)}


def gt_doc_indices_from_semids(
    gt_semids: Sequence[str], semid_to_doc_index: Dict[str, int]
) -> np.ndarray:
    missing = [semid for semid in gt_semids if semid not in semid_to_doc_index]
    if missing:
        raise ValueError(
            f"Could not map {len(missing)} GT semids to document indices. "
            f"First missing semid: {missing[0]}"
        )
    return np.asarray([semid_to_doc_index[semid] for semid in gt_semids], dtype=np.int64)


def search_topk_docs(
    doc_embs: np.ndarray,
    query_embs: np.ndarray,
    k_max: int,
    batch_size: int,
    search_extra: int,
    exclude_doc_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if faiss is None:
        raise ImportError("faiss is required for nearest-neighbor search.")
    if batch_size <= 0:
        raise ValueError("--search-batch-size must be > 0")

    n_docs, dim = doc_embs.shape
    if query_embs.shape[1] != dim:
        raise ValueError(
            f"Embedding dimension mismatch: docs={dim}, queries={query_embs.shape[1]}"
        )
    if k_max <= 0 or k_max >= n_docs:
        raise ValueError(f"k_max must be in [1, n_docs). Got k_max={k_max}, n_docs={n_docs}.")

    raw_k = k_max
    if exclude_doc_idx is not None:
        raw_k = min(n_docs, k_max + max(1, search_extra) + 1)

    doc_embs32 = np.ascontiguousarray(doc_embs, dtype=np.float32)
    query_embs32 = np.ascontiguousarray(query_embs, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs32)

    neighbors = np.empty((query_embs.shape[0], k_max), dtype=np.int32)
    scores = np.empty((query_embs.shape[0], k_max), dtype=np.float32)
    rows_refetched = 0
    gt_filtered_rows = 0

    for start in range(0, query_embs.shape[0], batch_size):
        end = min(start + batch_size, query_embs.shape[0])
        score_batch, idx_batch = index.search(query_embs32[start:end], raw_k)

        for row in range(end - start):
            global_row = start + row
            cand_idx = idx_batch[row]
            cand_scores = score_batch[row]

            if exclude_doc_idx is not None:
                keep = cand_idx != exclude_doc_idx[global_row]
                if np.any(~keep):
                    gt_filtered_rows += 1
                cand_idx = cand_idx[keep]
                cand_scores = cand_scores[keep]

                if cand_idx.size < k_max:
                    rows_refetched += 1
                    score_refetch, idx_refetch = index.search(
                        query_embs32[global_row : global_row + 1], n_docs
                    )
                    keep_refetch = idx_refetch[0] != exclude_doc_idx[global_row]
                    cand_idx = idx_refetch[0][keep_refetch]
                    cand_scores = score_refetch[0][keep_refetch]

            if cand_idx.size < k_max:
                raise RuntimeError(
                    f"Query row {global_row} has only {cand_idx.size} candidates; "
                    f"cannot satisfy K={k_max}."
                )

            neighbors[global_row] = cand_idx[:k_max].astype(np.int32)
            scores[global_row] = cand_scores[:k_max].astype(np.float32)

    meta = {
        "n_docs": int(n_docs),
        "n_queries": int(query_embs.shape[0]),
        "k_max": int(k_max),
        "search_k_raw": int(raw_k),
        "exclude_gt_doc": int(exclude_doc_idx is not None),
        "gt_filtered_rows": int(gt_filtered_rows),
        "rows_refetched": int(rows_refetched),
        "index_type": "IndexFlatIP",
        "approximate_search": 0,
    }
    return neighbors, scores, meta


def empty_level_dict(levels: int) -> Dict[str, float]:
    return {f"l{level}": 0.0 for level in range(1, levels + 1)}


def compute_prefix_metrics(
    rq_codes: np.ndarray,
    gt_codes: np.ndarray,
    neighbors: np.ndarray,
    k_values: Sequence[int],
    levels: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    k_max = neighbors.shape[1]
    recall_by_k = {k: empty_level_dict(levels) for k in k_values}
    hit_by_k = {k: empty_level_dict(levels) for k in k_values}

    for level in range(1, levels + 1):
        candidate_codes = rq_codes[neighbors[:, :k_max], :level]
        gt_prefix = gt_codes[:, None, :level]
        match = np.all(candidate_codes == gt_prefix, axis=2)
        for k in k_values:
            sub = match[:, :k]
            recall_by_k[k][f"l{level}"] = float(sub.mean())
            hit_by_k[k][f"l{level}"] = float(sub.any(axis=1).mean())

    recall_rows = []
    hit_rows = []
    for k in k_values:
        recall_row = {"method": "query_knn", "k": int(k)}
        hit_row = {"method": "query_knn", "k": int(k)}
        recall_row.update(recall_by_k[k])
        hit_row.update(hit_by_k[k])
        recall_rows.append(recall_row)
        hit_rows.append(hit_row)

    return recall_rows, hit_rows


def summarize_unique(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def compute_unique_l1_metrics(
    rq_codes: np.ndarray, neighbors: np.ndarray, k_values: Sequence[int]
) -> Tuple[List[Dict[str, float]], Dict[int, np.ndarray]]:
    n_queries = neighbors.shape[0]
    neighbor_l1 = rq_codes[neighbors, 0]
    rows = []
    distributions: Dict[int, np.ndarray] = {}

    for k in k_values:
        slice_l1 = neighbor_l1[:, :k]
        if k == 1:
            counts = np.ones(n_queries, dtype=np.int16)
        else:
            sorted_vals = np.sort(slice_l1, axis=1)
            counts = 1 + np.sum(sorted_vals[:, 1:] != sorted_vals[:, :-1], axis=1)
        stats = summarize_unique(counts.astype(np.float32))
        row = {"method": "query_knn", "k": int(k)}
        row.update(stats)
        rows.append(row)
        distributions[k] = counts

    return rows, distributions


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    del rng  # keep seed in args metadata for reproducibility; no random sampling here.

    k_values = parse_k_values(args.k_values)
    k_max = max(k_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test queries...")
    queries, gt_codes, gt_semids = load_test_queries(
        path=args.test_data,
        max_queries=args.max_queries,
        keep_query_prefix=args.keep_query_prefix,
    )
    print(f"  queries: {len(queries)}")

    print("Loading document embeddings...")
    doc_embs = np.load(args.doc_embeddings).astype(np.float32)
    doc_embs = l2_normalize(doc_embs)
    print(f"  doc embeddings: {doc_embs.shape}")

    query_embs = load_or_encode_query_embeddings(
        queries=queries,
        path=Path(args.query_embeddings),
        model_name=args.embedding_model,
        batch_size=args.query_batch_size,
        instruction=args.query_instruction,
        no_encode=args.no_encode,
        recompute=args.recompute_query_embeddings,
    ).astype(np.float32)
    if query_embs.shape[0] != len(queries):
        raise ValueError(
            f"Query embedding count mismatch: embeddings={query_embs.shape[0]}, "
            f"queries={len(queries)}"
        )
    query_embs = l2_normalize(query_embs)
    print(f"  query embeddings: {query_embs.shape}")

    print("Loading RQ codes...")
    rq_codes = load_rq_codes(Path(args.rq_codes), Path(args.idx_to_rqid))
    if rq_codes.shape[0] != doc_embs.shape[0]:
        raise ValueError(
            f"Doc count mismatch: embeddings={doc_embs.shape[0]}, rq_codes={rq_codes.shape[0]}"
        )
    levels = min(args.levels, rq_codes.shape[1], gt_codes.shape[1])
    gt_codes = gt_codes[:, :levels]
    print(f"  rq_codes: {rq_codes.shape}, evaluating levels=1..{levels}")

    exclude_doc_idx = None
    if args.exclude_gt_doc:
        print("Building GT semid -> document index map for --exclude-gt-doc...")
        semid_to_doc_index = build_semid_to_doc_index(args)
        exclude_doc_idx = gt_doc_indices_from_semids(gt_semids, semid_to_doc_index)

    print(f"Searching query top-{k_max} documents...")
    neighbors, scores, search_meta = search_topk_docs(
        doc_embs=doc_embs,
        query_embs=query_embs,
        k_max=k_max,
        batch_size=args.search_batch_size,
        search_extra=args.search_extra,
        exclude_doc_idx=exclude_doc_idx,
    )
    print(f"  neighbors: {neighbors.shape}, scores: {scores.shape}")

    print("Computing prefix metrics...")
    recall_rows, hit_rows = compute_prefix_metrics(
        rq_codes=rq_codes,
        gt_codes=gt_codes,
        neighbors=neighbors,
        k_values=k_values,
        levels=levels,
    )
    unique_rows, _ = compute_unique_l1_metrics(
        rq_codes=rq_codes,
        neighbors=neighbors,
        k_values=k_values,
    )

    level_headers = [f"l{level}" for level in range(1, levels + 1)]
    recall_csv = output_dir / "query_prefix_recall_at_k.csv"
    hit_csv = output_dir / "query_prefix_hit_at_k.csv"
    unique_csv = output_dir / "query_unique_l1_at_k.csv"

    write_csv(recall_csv, recall_rows, ["method", "k", *level_headers])
    write_csv(hit_csv, hit_rows, ["method", "k", *level_headers])
    write_csv(unique_csv, unique_rows, ["method", "k", "mean", "median", "q25", "q75"])

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "k_values": k_values,
        "n_queries": len(queries),
        "doc_embeddings_shape": list(doc_embs.shape),
        "query_embeddings_shape": list(query_embs.shape),
        "rq_codes_shape": list(rq_codes.shape),
        "evaluated_levels": levels,
        "search": search_meta,
        "query_prefix_recall_at_k": recall_rows,
        "query_prefix_hit_at_k": hit_rows,
        "query_unique_l1_at_k": unique_rows,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Done.")
    print(f"PrefixRecall CSV: {recall_csv}")
    print(f"PrefixHit CSV: {hit_csv}")
    print(f"UniqueL1 CSV: {unique_csv}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
