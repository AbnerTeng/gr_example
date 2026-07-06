import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from src.analyze_query_rq_prefix import (
    build_semid_to_doc_index,
    gt_doc_indices_from_semids,
    l2_normalize,
    load_or_encode_query_embeddings,
    load_rq_codes,
    load_test_queries,
    search_topk_docs,
)
from src.analyze_rq_ranking_consistency import make_bin_edges, to_json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate P(candidate doc shares GT RQ prefix | query-doc cosine "
            "similarity bin) over query embedding top-K retrieved documents."
        )
    )
    parser.add_argument("--doc-embeddings", type=str, default="data/doc_embeddings.npy")
    parser.add_argument("--query-embeddings", type=str, default="data/test_query_embeddings.npy")
    parser.add_argument("--rq-codes", type=str, default="data/rq_codes.npy")
    parser.add_argument("--idx-to-rqid", type=str, default="data/idx_to_rqid.json")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/query_rq_ranking_consistency",
    )
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--search-batch-size", type=int, default=8192)
    parser.add_argument(
        "--knn-k",
        type=int,
        default=100,
        help="Number of retrieved documents per query used as query-doc pairs.",
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
    parser.add_argument("--bin-start", type=float, default=0.0)
    parser.add_argument("--bin-end", type=float, default=1.0)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument(
        "--n-random-baselines",
        type=int,
        default=1,
        help="Number of shuffled-RQ random-code baselines. Use 0 to skip.",
    )
    parser.add_argument(
        "--plot-min-pairs",
        type=int,
        default=100,
        help="Minimum pairs per bin for plotting probability curves.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--msmarco-train", type=str, default="data/msmarco/train.jsonl")
    parser.add_argument("--msmarco-valid", type=str, default="data/msmarco/valid.jsonl")
    parser.add_argument("--msmarco-test", type=str, default="data/msmarco/test.jsonl")
    return parser.parse_args()


def empty_accumulators(n_bins: int, levels: int) -> Dict[str, np.ndarray]:
    return {
        "counts": np.zeros(n_bins, dtype=np.int64),
        "sim_sums": np.zeros(n_bins, dtype=np.float64),
        "shared_sums": np.zeros((levels, n_bins), dtype=np.float64),
        "outside_low": np.zeros(1, dtype=np.int64),
        "outside_high": np.zeros(1, dtype=np.int64),
    }


def compute_accumulators(
    similarities: np.ndarray,
    query_indices: np.ndarray,
    candidate_doc_indices: np.ndarray,
    rq_codes: np.ndarray,
    gt_codes: np.ndarray,
    bin_edges: np.ndarray,
    levels: int,
) -> Dict[str, np.ndarray]:
    n_bins = len(bin_edges) - 1
    acc = empty_accumulators(n_bins, levels)
    if similarities.size == 0:
        return acc

    bin_idx = np.searchsorted(bin_edges, similarities, side="right") - 1
    bin_idx[similarities == bin_edges[-1]] = n_bins - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    acc["outside_low"][0] = int(np.sum(bin_idx < 0))
    acc["outside_high"][0] = int(np.sum(bin_idx >= n_bins))

    if not np.any(valid):
        return acc

    valid_bins = bin_idx[valid]
    valid_sims = similarities[valid].astype(np.float64)
    acc["counts"] += np.bincount(valid_bins, minlength=n_bins).astype(np.int64)
    acc["sim_sums"] += np.bincount(
        valid_bins, weights=valid_sims, minlength=n_bins
    ).astype(np.float64)

    shared = np.ones(similarities.shape[0], dtype=bool)
    for level in range(levels):
        shared &= (
            rq_codes[candidate_doc_indices, level]
            == gt_codes[query_indices, level]
        )
        acc["shared_sums"][level] += np.bincount(
            valid_bins,
            weights=shared[valid].astype(np.float64),
            minlength=n_bins,
        ).astype(np.float64)

    return acc


def rows_from_accumulators(
    acc: Dict[str, np.ndarray],
    method: str,
    bin_edges: np.ndarray,
    levels: int,
) -> List[Dict[str, float]]:
    rows = []
    counts = acc["counts"]
    for i in range(len(bin_edges) - 1):
        n = int(counts[i])
        row = {
            "method": method,
            "bin_left": float(bin_edges[i]),
            "bin_right": float(bin_edges[i + 1]),
            "bin_mid": float((bin_edges[i] + bin_edges[i + 1]) / 2.0),
            "n_pairs": n,
            "mean_similarity": float(acc["sim_sums"][i] / n) if n else float("nan"),
        }
        for level in range(levels):
            row[f"l{level + 1}"] = (
                float(acc["shared_sums"][level, i] / n) if n else float("nan")
            )
        rows.append(row)
    return rows


def evaluate_method(
    method: str,
    similarities: np.ndarray,
    query_indices: np.ndarray,
    candidate_doc_indices: np.ndarray,
    rq_codes: np.ndarray,
    gt_codes: np.ndarray,
    bin_edges: np.ndarray,
    levels: int,
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    acc = compute_accumulators(
        similarities=similarities,
        query_indices=query_indices,
        candidate_doc_indices=candidate_doc_indices,
        rq_codes=rq_codes,
        gt_codes=gt_codes,
        bin_edges=bin_edges,
        levels=levels,
    )
    rows = rows_from_accumulators(acc, method, bin_edges, levels)
    meta = {
        "n_pairs": int(similarities.shape[0]),
        "outside_low": int(acc["outside_low"][0]),
        "outside_high": int(acc["outside_high"][0]),
    }
    return rows, meta


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_probability_curves(
    rows: List[Dict[str, float]],
    levels: int,
    out_path: Path,
    min_pairs: int,
) -> None:
    method_styles = {
        "query_knn": "-",
        "random_code": "--",
    }
    plot_rows = [
        row
        for row in rows
        if row["method"] in method_styles and int(row["n_pairs"]) >= min_pairs
    ]
    if not plot_rows:
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for level in range(1, levels + 1):
        color = colors[(level - 1) % len(colors)]
        for method, linestyle in method_styles.items():
            method_rows = sorted(
                [row for row in plot_rows if row["method"] == method],
                key=lambda row: float(row["bin_mid"]),
            )
            if not method_rows:
                continue
            ax.plot(
                [row["bin_mid"] for row in method_rows],
                [row[f"l{level}"] for row in method_rows],
                linestyle=linestyle,
                marker="o" if method == "query_knn" else None,
                linewidth=1.8,
                markersize=3,
                color=color,
                label=f"{method} L{level}",
            )

    ax.set_xlabel("Query-Document Cosine Similarity")
    ax.set_ylabel("P(candidate shares GT RQ prefix)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Query Ranking Consistency by Similarity")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.knn_k <= 0:
        raise ValueError("--knn-k must be > 0")
    if args.n_random_baselines < 0:
        raise ValueError("--n-random-baselines must be >= 0")
    if args.plot_min_pairs < 0:
        raise ValueError("--plot-min-pairs must be >= 0")

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test queries...")
    queries, parsed_gt_codes, gt_semids = load_test_queries(
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
    levels = min(args.levels, rq_codes.shape[1], parsed_gt_codes.shape[1])
    print(f"  rq_codes: {rq_codes.shape}, evaluating levels=1..{levels}")

    print("Building GT semid -> document index map...")
    semid_to_doc_index = build_semid_to_doc_index(args)
    gt_doc_indices = gt_doc_indices_from_semids(gt_semids, semid_to_doc_index)

    parsed_gt_codes = parsed_gt_codes[:, :levels]
    mapped_gt_codes = rq_codes[gt_doc_indices, :levels]
    mismatch = int(np.sum(np.any(parsed_gt_codes != mapped_gt_codes, axis=1)))
    if mismatch:
        raise ValueError(
            f"GT RQ code mismatch for {mismatch} queries between test data and rq_codes."
        )

    exclude_doc_idx = gt_doc_indices if args.exclude_gt_doc else None

    print(f"Searching query top-{args.knn_k} documents...")
    neighbors, scores, search_meta = search_topk_docs(
        doc_embs=doc_embs,
        query_embs=query_embs,
        k_max=args.knn_k,
        batch_size=args.search_batch_size,
        search_extra=args.search_extra,
        exclude_doc_idx=exclude_doc_idx,
    )
    print(f"  neighbors: {neighbors.shape}, scores: {scores.shape}")

    query_indices = np.repeat(np.arange(neighbors.shape[0], dtype=np.int64), args.knn_k)
    candidate_doc_indices = neighbors.reshape(-1).astype(np.int64)
    similarities = scores.reshape(-1).astype(np.float32)

    bin_edges = make_bin_edges(args.bin_start, args.bin_end, args.bin_width)
    print(
        f"Using {len(bin_edges) - 1} similarity bins: "
        f"[{bin_edges[0]:.3f}, {bin_edges[-1]:.3f}], width={args.bin_width}"
    )

    all_rows: List[Dict[str, float]] = []
    method_meta: Dict[str, Dict[str, int]] = {}

    print("Evaluating query_knn...")
    rows, meta = evaluate_method(
        method="query_knn",
        similarities=similarities,
        query_indices=query_indices,
        candidate_doc_indices=candidate_doc_indices,
        rq_codes=rq_codes,
        gt_codes=mapped_gt_codes,
        bin_edges=bin_edges,
        levels=levels,
    )
    all_rows.extend(rows)
    method_meta["query_knn"] = meta

    random_baseline_runs = []
    for run in range(args.n_random_baselines):
        method = "random_code" if args.n_random_baselines == 1 else f"random_code_{run + 1}"
        print(f"Evaluating {method} baseline...")
        perm = rng.permutation(rq_codes.shape[0])
        rq_random = rq_codes[perm]
        random_gt_codes = rq_random[gt_doc_indices, :levels]
        rows, meta = evaluate_method(
            method=method,
            similarities=similarities,
            query_indices=query_indices,
            candidate_doc_indices=candidate_doc_indices,
            rq_codes=rq_random,
            gt_codes=random_gt_codes,
            bin_edges=bin_edges,
            levels=levels,
        )
        all_rows.extend(rows)
        method_meta[method] = meta
        random_baseline_runs.append({"run": run + 1, "method": method})

    level_headers = [f"l{level}" for level in range(1, levels + 1)]
    csv_path = output_dir / "shared_prefix_by_query_similarity.csv"
    write_csv(
        csv_path,
        all_rows,
        ["method", "bin_left", "bin_right", "bin_mid", "n_pairs", "mean_similarity", *level_headers],
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "n_queries": len(queries),
        "doc_embeddings_shape": list(doc_embs.shape),
        "query_embeddings_shape": list(query_embs.shape),
        "rq_codes_shape": list(rq_codes.shape),
        "evaluated_levels": levels,
        "bin_edges": [float(x) for x in bin_edges],
        "pair_source": "query_knn_docs",
        "search": search_meta,
        "method_meta": method_meta,
        "random_baseline_runs": random_baseline_runs,
        "shared_prefix_by_query_similarity": all_rows,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2)

    if args.no_plots:
        print("Skipping plots (--no-plots).")
    else:
        if plt is None:
            print("matplotlib not installed, skipping plot generation.")
        else:
            plot_path = output_dir / "query_shared_prefix_probability.png"
            plot_probability_curves(
                rows=all_rows,
                levels=levels,
                out_path=plot_path,
                min_pairs=args.plot_min_pairs,
            )
            print(f"Saved plot: {plot_path}")

    print("Done.")
    print(f"CSV: {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
