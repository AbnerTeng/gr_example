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
    load_test_queries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether query-document embedding similarity ranks each query's "
            "gold document near the top of the full document collection."
        )
    )
    parser.add_argument("--doc-embeddings", type=str, default="data/doc_embeddings.npy")
    parser.add_argument("--query-embeddings", type=str, default="data/test_query_embeddings.npy")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/query_gold_rank")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument(
        "--rank-batch-size",
        type=int,
        default=128,
        help="Number of queries per exact full-corpus similarity batch.",
    )
    parser.add_argument(
        "--recall-k-values",
        type=str,
        default="1,5,10,20,50,100,200,500,1000",
        help="Comma-separated K values for Gold Recall@K.",
    )
    parser.add_argument(
        "--top-n-non-gold",
        type=int,
        default=10,
        help="Number of highest-scoring non-gold documents used for similarity summaries.",
    )
    parser.add_argument(
        "--random-docs-per-query",
        type=int,
        default=1,
        help="Number of non-gold random documents sampled per query for the random baseline.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
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


def summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def sample_random_doc_indices(
    n_queries: int,
    n_docs: int,
    gt_doc_indices: np.ndarray,
    random_docs_per_query: int,
    rng: np.random.Generator,
) -> np.ndarray:
    random_idx = rng.integers(
        0, n_docs, size=(n_queries, random_docs_per_query), dtype=np.int64
    )
    gt = gt_doc_indices[:, None]
    random_idx = np.where(random_idx == gt, (random_idx + 1) % n_docs, random_idx)
    return random_idx


def compute_gold_rank_metrics(
    doc_embs: np.ndarray,
    query_embs: np.ndarray,
    gt_doc_indices: np.ndarray,
    random_doc_indices: np.ndarray,
    batch_size: int,
    top_n_non_gold: int,
) -> Dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("--rank-batch-size must be > 0")
    if doc_embs.ndim != 2 or query_embs.ndim != 2:
        raise ValueError("Expected 2D document and query embedding arrays.")
    if doc_embs.shape[1] != query_embs.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: docs={doc_embs.shape[1]}, "
            f"queries={query_embs.shape[1]}"
        )
    if top_n_non_gold <= 0:
        raise ValueError("--top-n-non-gold must be > 0")
    if top_n_non_gold >= doc_embs.shape[0]:
        raise ValueError("--top-n-non-gold must be smaller than the document count.")

    n_queries = query_embs.shape[0]
    n_docs = doc_embs.shape[0]
    doc_embs_t = np.ascontiguousarray(doc_embs.T, dtype=np.float32)

    ranks = np.empty(n_queries, dtype=np.int64)
    gold_scores = np.empty(n_queries, dtype=np.float32)
    top1_scores = np.empty(n_queries, dtype=np.float32)
    top1_doc_indices = np.empty(n_queries, dtype=np.int64)
    top_non_gold_scores = np.empty((n_queries, top_n_non_gold), dtype=np.float32)
    random_scores = np.empty(random_doc_indices.shape, dtype=np.float32)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_queries = np.ascontiguousarray(query_embs[start:end], dtype=np.float32)
        sims = batch_queries @ doc_embs_t

        local_rows = np.arange(end - start)
        gt_idx = gt_doc_indices[start:end]
        gold_batch = sims[local_rows, gt_idx].astype(np.float32)
        gold_scores[start:end] = gold_batch
        ranks[start:end] = 1 + np.sum(sims > gold_batch[:, None], axis=1)

        top1_idx = np.argmax(sims, axis=1)
        top1_doc_indices[start:end] = top1_idx.astype(np.int64)
        top1_scores[start:end] = sims[local_rows, top1_idx].astype(np.float32)

        sims_without_gold = sims.copy()
        sims_without_gold[local_rows, gt_idx] = -np.inf
        top_scores = np.partition(sims_without_gold, -top_n_non_gold, axis=1)[
            :, -top_n_non_gold:
        ]
        top_non_gold_scores[start:end] = top_scores.astype(np.float32)

        random_idx = random_doc_indices[start:end]
        random_scores[start:end] = sims[
            local_rows[:, None],
            random_idx,
        ].astype(np.float32)

    return {
        "gold_rank": ranks,
        "gold_similarity": gold_scores,
        "top1_doc_index": top1_doc_indices,
        "top1_similarity": top1_scores,
        "top_non_gold_similarity": top_non_gold_scores,
        "random_similarity": random_scores,
    }


def make_recall_rows(ranks: np.ndarray, k_values: Sequence[int]) -> List[Dict[str, float]]:
    rows = []
    for k in k_values:
        rows.append({"k": int(k), "gold_recall_at_k": float(np.mean(ranks <= k))})
    return rows


def make_histogram_rows(ranks: np.ndarray) -> List[Dict[str, float]]:
    buckets = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-500", 101, 500),
        (">500", 501, None),
    ]
    rows = []
    n = ranks.size
    for label, low, high in buckets:
        if high is None:
            mask = ranks >= low
        else:
            mask = (ranks >= low) & (ranks <= high)
        count = int(mask.sum())
        rows.append(
            {
                "bucket": label,
                "count": count,
                "fraction": float(count / n) if n else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_query_csv(
    path: Path,
    gt_semids: Sequence[str],
    gt_doc_indices: np.ndarray,
    metrics: Dict[str, np.ndarray],
) -> None:
    top_non_gold_mean = metrics["top_non_gold_similarity"].mean(axis=1)
    random_mean = metrics["random_similarity"].mean(axis=1)
    gaps = metrics["top1_similarity"] - metrics["gold_similarity"]

    fieldnames = [
        "query_index",
        "gt_semid",
        "gt_doc_index",
        "gold_rank",
        "gold_similarity",
        "top1_doc_index",
        "top1_similarity",
        "top1_is_gold",
        "top1_minus_gold_gap",
        "top_non_gold_mean_similarity",
        "random_mean_similarity",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, semid in enumerate(gt_semids):
            writer.writerow(
                {
                    "query_index": i,
                    "gt_semid": semid,
                    "gt_doc_index": int(gt_doc_indices[i]),
                    "gold_rank": int(metrics["gold_rank"][i]),
                    "gold_similarity": float(metrics["gold_similarity"][i]),
                    "top1_doc_index": int(metrics["top1_doc_index"][i]),
                    "top1_similarity": float(metrics["top1_similarity"][i]),
                    "top1_is_gold": int(metrics["top1_doc_index"][i] == gt_doc_indices[i]),
                    "top1_minus_gold_gap": float(gaps[i]),
                    "top_non_gold_mean_similarity": float(top_non_gold_mean[i]),
                    "random_mean_similarity": float(random_mean[i]),
                }
            )


def plot_gold_rank_cdf(
    recall_rows: List[Dict[str, float]], out_path: Path, n_docs: int
) -> None:
    x = np.asarray([row["k"] for row in recall_rows], dtype=np.float64)
    y = np.asarray([row["gold_recall_at_k"] for row in recall_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2)
    for ref in [10, 50, 100]:
        if 1 <= ref <= n_docs:
            ax.axvline(ref, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.text(ref, 0.02, f"K={ref}", rotation=90, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Rank Threshold K")
    ax.set_ylabel("Gold Recall@K")
    ax.set_title("Gold Rank CDF Under Query-Document Embedding Similarity")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gold_rank_histogram(hist_rows: List[Dict[str, float]], out_path: Path) -> None:
    labels = [row["bucket"] for row in hist_rows]
    fractions = [row["fraction"] for row in hist_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, fractions)
    for bar, frac in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{frac:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0.0, max(fractions) * 1.18 if fractions else 1.0)
    ax.set_xlabel("Gold Rank Bucket")
    ax.set_ylabel("Query Fraction")
    ax.set_title("Gold Rank Histogram")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_similarity_distributions(
    random_scores: np.ndarray,
    gold_scores: np.ndarray,
    top_non_gold_scores: np.ndarray,
    top1_scores: np.ndarray,
    out_path: Path,
    top_n_non_gold: int,
) -> None:
    distributions = [
        random_scores.reshape(-1),
        gold_scores,
        top_non_gold_scores.reshape(-1),
        top1_scores,
    ]
    labels = [
        "Random doc",
        "Gold doc",
        f"Top-{top_n_non_gold} non-gold docs",
        "Top-1 doc",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(distributions, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Query-Document Cosine Similarity")
    ax.set_title("Similarity Score Distributions")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def to_json_safe(value):
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if args.random_docs_per_query <= 0:
        raise ValueError("--random-docs-per-query must be > 0")

    rng = np.random.default_rng(args.seed)
    k_values = parse_k_values(args.recall_k_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test queries...")
    queries, _, gt_semids = load_test_queries(
        path=args.test_data,
        max_queries=args.max_queries,
        keep_query_prefix=args.keep_query_prefix,
    )
    print(f"  queries: {len(queries)}")

    print("Loading document embeddings...")
    doc_embs = np.load(args.doc_embeddings).astype(np.float32)
    doc_embs = l2_normalize(doc_embs)
    print(f"  doc embeddings: {doc_embs.shape}")
    n_docs = doc_embs.shape[0]
    too_large = [k for k in k_values if k > n_docs]
    if too_large:
        raise ValueError(f"Recall K cannot exceed n_docs={n_docs}. Got {too_large}")

    print("Loading or encoding query embeddings...")
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

    print("Building GT semid -> document index map...")
    semid_to_doc_index = build_semid_to_doc_index(args)
    gt_doc_indices = gt_doc_indices_from_semids(gt_semids, semid_to_doc_index)

    print("Sampling random non-gold documents...")
    random_doc_indices = sample_random_doc_indices(
        n_queries=len(queries),
        n_docs=n_docs,
        gt_doc_indices=gt_doc_indices,
        random_docs_per_query=args.random_docs_per_query,
        rng=rng,
    )

    print("Computing exact gold ranks over the full document collection...")
    metrics = compute_gold_rank_metrics(
        doc_embs=doc_embs,
        query_embs=query_embs,
        gt_doc_indices=gt_doc_indices,
        random_doc_indices=random_doc_indices,
        batch_size=args.rank_batch_size,
        top_n_non_gold=args.top_n_non_gold,
    )

    ranks = metrics["gold_rank"]
    gold_scores = metrics["gold_similarity"]
    top1_scores = metrics["top1_similarity"]
    top_non_gold_scores = metrics["top_non_gold_similarity"]
    random_scores = metrics["random_similarity"]
    gaps = top1_scores - gold_scores

    recall_rows = make_recall_rows(ranks, k_values)
    hist_rows = make_histogram_rows(ranks)
    summary_rows = [
        {"metric": "gold_rank", **summarize(ranks.astype(np.float64))},
        {"metric": "gold_similarity", **summarize(gold_scores)},
        {"metric": "top1_similarity", **summarize(top1_scores)},
        {
            "metric": f"top{args.top_n_non_gold}_non_gold_similarity",
            **summarize(top_non_gold_scores.reshape(-1)),
        },
        {"metric": "random_similarity", **summarize(random_scores.reshape(-1))},
        {"metric": "top1_minus_gold_gap", **summarize(gaps)},
    ]
    gap_summary = {
        "mean_gap": float(gaps.mean()),
        "median_gap": float(np.median(gaps)),
        "fraction_gap_lt_0_01": float(np.mean(gaps < 0.01)),
        "fraction_gap_lt_0_05": float(np.mean(gaps < 0.05)),
    }

    per_query_csv = output_dir / "per_query_gold_rank.csv"
    recall_csv = output_dir / "gold_recall_at_k.csv"
    hist_csv = output_dir / "gold_rank_histogram.csv"
    summary_csv = output_dir / "summary_stats.csv"
    json_path = output_dir / "results.json"

    write_per_query_csv(per_query_csv, gt_semids, gt_doc_indices, metrics)
    write_csv(recall_csv, recall_rows, ["k", "gold_recall_at_k"])
    write_csv(hist_csv, hist_rows, ["bucket", "count", "fraction"])
    write_csv(
        summary_csv,
        summary_rows,
        [
            "metric",
            "n",
            "mean",
            "median",
            "std",
            "min",
            "p10",
            "q25",
            "q75",
            "p90",
            "p95",
            "max",
        ],
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "n_queries": len(queries),
        "n_docs": int(n_docs),
        "doc_embeddings_shape": list(doc_embs.shape),
        "query_embeddings_shape": list(query_embs.shape),
        "rank_summary": summary_rows[0],
        "similarity_summaries": summary_rows[1:5],
        "gap_summary": gap_summary,
        "gold_recall_at_k": recall_rows,
        "gold_rank_histogram": hist_rows,
        "outputs": {
            "per_query_gold_rank": str(per_query_csv),
            "gold_recall_at_k": str(recall_csv),
            "gold_rank_histogram": str(hist_csv),
            "summary_stats": str(summary_csv),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2)

    if args.no_plots:
        print("Skipping plots (--no-plots).")
    elif plt is None:
        print("matplotlib not installed, skipping plot generation.")
    else:
        cdf_path = output_dir / "gold_rank_cdf.png"
        hist_path = output_dir / "gold_rank_histogram.png"
        sim_path = output_dir / "similarity_distributions.png"
        plot_gold_rank_cdf(recall_rows, cdf_path, n_docs=n_docs)
        plot_gold_rank_histogram(hist_rows, hist_path)
        plot_similarity_distributions(
            random_scores=random_scores,
            gold_scores=gold_scores,
            top_non_gold_scores=top_non_gold_scores,
            top1_scores=top1_scores,
            out_path=sim_path,
            top_n_non_gold=args.top_n_non_gold,
        )
        print(f"Saved plots: {cdf_path}, {hist_path}, {sim_path}")

    print("Done.")
    print(f"Per-query CSV: {per_query_csv}")
    print(f"Recall CSV: {recall_csv}")
    print(f"Histogram CSV: {hist_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
