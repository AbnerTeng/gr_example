import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


RQ_PATTERN = re.compile(r"<r\d+_(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate P(shared RQ prefix | embedding cosine similarity bin) "
            "within embedding kNN pairs."
        )
    )
    parser.add_argument("--doc-embeddings", type=str, default="data/doc_embeddings.npy")
    parser.add_argument("--rq-codes", type=str, default="data/rq_codes.npy")
    parser.add_argument("--idx-to-rqid", type=str, default="data/idx_to_rqid.json")
    parser.add_argument("--output-dir", type=str, default="outputs/rq_ranking_consistency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--knn-k",
        type=int,
        default=100,
        help="Number of nearest neighbors per anchor.",
    )
    parser.add_argument(
        "--anchor-docs",
        type=int,
        default=-1,
        help="Number of kNN anchor docs. -1 means all docs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size for FAISS kNN search.",
    )
    parser.add_argument(
        "--search-extra",
        type=int,
        default=32,
        help="Search K+extra neighbors before self-exclusion.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Maximum prefix levels to evaluate. Uses min(levels, rq_code_width).",
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
    return parser.parse_args()


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


def make_bin_edges(start: float, end: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("--bin-width must be > 0")
    if end <= start:
        raise ValueError("--bin-end must be greater than --bin-start")
    n_bins = int(np.ceil((end - start) / width))
    edges = start + np.arange(n_bins + 1, dtype=np.float64) * width
    edges[-1] = end
    return edges


def select_anchor_indices(
    n_docs: int, anchor_docs: int, rng: np.random.Generator
) -> np.ndarray:
    if anchor_docs <= 0 or anchor_docs >= n_docs:
        return np.arange(n_docs, dtype=np.int64)
    return np.asarray(rng.choice(n_docs, size=anchor_docs, replace=False), dtype=np.int64)


def search_knn_pairs(
    embs: np.ndarray,
    anchor_idx: np.ndarray,
    knn_k: int,
    batch_size: int,
    search_extra: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if faiss is None:
        raise ImportError("faiss is required for nearest-neighbor search.")
    if knn_k <= 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.asarray([], dtype=np.float32),
            {"n_docs": int(embs.shape[0]), "n_anchors": int(len(anchor_idx)), "knn_k": 0},
        )

    n_docs, dim = embs.shape
    if knn_k >= n_docs:
        raise ValueError(f"--knn-k must be < n_docs. Got knn_k={knn_k}, n_docs={n_docs}.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    raw_k = min(n_docs, knn_k + max(1, search_extra) + 1)
    embs32 = np.ascontiguousarray(embs, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(embs32)

    total_pairs = len(anchor_idx) * knn_k
    pairs = np.empty((total_pairs, 2), dtype=np.int32)
    sims = np.empty(total_pairs, dtype=np.float32)
    self_hit_rows = 0
    rows_refetched = 0
    out_pos = 0

    for start in range(0, len(anchor_idx), batch_size):
        end = min(start + batch_size, len(anchor_idx))
        anchors_batch = anchor_idx[start:end]
        score_batch, idx_batch = index.search(embs32[anchors_batch], raw_k)

        for row, src in enumerate(anchors_batch):
            candidates = idx_batch[row]
            scores = score_batch[row]
            keep = candidates != src
            if np.any(~keep):
                self_hit_rows += 1
            filtered_idx = candidates[keep]
            filtered_scores = scores[keep]

            if filtered_idx.size < knn_k:
                rows_refetched += 1
                score_refetch, idx_refetch = index.search(
                    embs32[src : src + 1], n_docs
                )
                keep_refetch = idx_refetch[0] != src
                filtered_idx = idx_refetch[0][keep_refetch]
                filtered_scores = score_refetch[0][keep_refetch]

            if filtered_idx.size < knn_k:
                raise RuntimeError(
                    f"Anchor {src} has only {filtered_idx.size} neighbors after self exclusion; "
                    f"cannot satisfy K={knn_k}."
                )

            take_idx = filtered_idx[:knn_k]
            take_scores = filtered_scores[:knn_k]
            block_end = out_pos + knn_k
            pairs[out_pos:block_end, 0] = int(src)
            pairs[out_pos:block_end, 1] = take_idx.astype(np.int32)
            sims[out_pos:block_end] = take_scores.astype(np.float32)
            out_pos = block_end

    meta = {
        "n_docs": int(n_docs),
        "n_anchors": int(len(anchor_idx)),
        "knn_k": int(knn_k),
        "n_pairs": int(total_pairs),
        "search_k_raw": int(raw_k),
        "self_hit_rows": int(self_hit_rows),
        "rows_refetched": int(rows_refetched),
        "index_type": "IndexFlatIP",
        "approximate_search": 0,
    }
    return pairs, sims, meta


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
    pairs: np.ndarray,
    rq_codes: np.ndarray,
    bin_edges: np.ndarray,
    levels: int,
) -> Dict[str, np.ndarray]:
    n_bins = len(bin_edges) - 1
    acc = empty_accumulators(n_bins, levels)
    if pairs.size == 0:
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

    left = pairs[:, 0].astype(np.int64)
    right = pairs[:, 1].astype(np.int64)
    shared = np.ones(pairs.shape[0], dtype=bool)
    for level in range(levels):
        shared &= rq_codes[left, level] == rq_codes[right, level]
        shared_valid = shared[valid].astype(np.float64)
        acc["shared_sums"][level] += np.bincount(
            valid_bins, weights=shared_valid, minlength=n_bins
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
    rq_codes: np.ndarray,
    pairs: np.ndarray,
    similarities: np.ndarray,
    bin_edges: np.ndarray,
    levels: int,
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    acc = compute_accumulators(
        similarities=similarities,
        pairs=pairs,
        rq_codes=rq_codes,
        bin_edges=bin_edges,
        levels=levels,
    )
    rows = rows_from_accumulators(acc, method, bin_edges, levels)
    meta = {
        "n_pairs": int(pairs.shape[0]),
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


def to_json_safe(value):
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_probability_curves(
    rows: List[Dict[str, float]],
    levels: int,
    out_path: Path,
    min_pairs: int,
) -> None:
    method_styles = {
        "vanilla_rq": "-",
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
                marker="o" if method == "vanilla_rq" else None,
                linewidth=1.8,
                markersize=3,
                color=color,
                label=f"{method} L{level}",
            )

    ax.set_xlabel("Embedding Cosine Similarity")
    ax.set_ylabel("P(shared RQ prefix)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Ranking Consistency: Shared Prefix Probability by Similarity")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.knn_k <= 0:
        raise ValueError("--knn-k must be > 0 for kNN-only ranking consistency.")
    if args.n_random_baselines < 0:
        raise ValueError("--n-random-baselines must be >= 0")
    if args.plot_min_pairs < 0:
        raise ValueError("--plot-min-pairs must be >= 0")

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    embs = np.load(args.doc_embeddings).astype(np.float32)
    embs = l2_normalize(embs)
    print(f"  embeddings: {embs.shape}")

    print("Loading RQ codes...")
    rq_codes = load_rq_codes(Path(args.rq_codes), Path(args.idx_to_rqid))
    if rq_codes.shape[0] != embs.shape[0]:
        raise ValueError(
            f"Doc count mismatch: embeddings={embs.shape[0]}, rq_codes={rq_codes.shape[0]}"
        )
    levels = min(args.levels, rq_codes.shape[1])
    print(f"  rq_codes: {rq_codes.shape}, evaluating levels=1..{levels}")

    bin_edges = make_bin_edges(args.bin_start, args.bin_end, args.bin_width)
    print(
        f"Using {len(bin_edges) - 1} similarity bins: "
        f"[{bin_edges[0]:.3f}, {bin_edges[-1]:.3f}], width={args.bin_width}"
    )

    anchor_idx = select_anchor_indices(embs.shape[0], args.anchor_docs, rng)
    print(f"Searching top-{args.knn_k} kNN pairs for {len(anchor_idx)} anchors...")
    knn_pairs, knn_sims, knn_meta = search_knn_pairs(
        embs=embs,
        anchor_idx=anchor_idx,
        knn_k=args.knn_k,
        batch_size=args.batch_size,
        search_extra=args.search_extra,
    )
    print(
        f"  knn_pairs: {knn_pairs.shape[0]}, "
        f"similarity range=({knn_sims.min():.4f}, {knn_sims.max():.4f})"
        if knn_sims.size
        else "  knn_pairs: 0"
    )

    all_rows: List[Dict[str, float]] = []
    method_meta: Dict[str, Dict[str, int]] = {}

    print("Evaluating vanilla RQ...")
    rows, meta = evaluate_method(
        method="vanilla_rq",
        rq_codes=rq_codes,
        pairs=knn_pairs,
        similarities=knn_sims,
        bin_edges=bin_edges,
        levels=levels,
    )
    all_rows.extend(rows)
    method_meta["vanilla_rq"] = meta

    random_baseline_runs = []
    for run in range(args.n_random_baselines):
        method = "random_code" if args.n_random_baselines == 1 else f"random_code_{run + 1}"
        print(f"Evaluating {method} baseline...")
        perm = rng.permutation(rq_codes.shape[0])
        rq_random = rq_codes[perm]
        rows, meta = evaluate_method(
            method=method,
            rq_codes=rq_random,
            pairs=knn_pairs,
            similarities=knn_sims,
            bin_edges=bin_edges,
            levels=levels,
        )
        all_rows.extend(rows)
        method_meta[method] = meta
        random_baseline_runs.append({"run": run + 1, "method": method})

    level_headers = [f"l{level}" for level in range(1, levels + 1)]
    csv_path = output_dir / "shared_prefix_by_similarity.csv"
    fieldnames = [
        "method",
        "bin_left",
        "bin_right",
        "bin_mid",
        "n_pairs",
        "mean_similarity",
        *level_headers,
    ]
    write_csv(csv_path, all_rows, fieldnames)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "embeddings_shape": list(embs.shape),
        "rq_codes_shape": list(rq_codes.shape),
        "evaluated_levels": levels,
        "bin_edges": [float(x) for x in bin_edges],
        "pair_source": "knn_pairs",
        "knn_search": knn_meta,
        "method_meta": method_meta,
        "random_baseline_runs": random_baseline_runs,
        "shared_prefix_by_similarity": all_rows,
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
            prob_path = output_dir / "shared_prefix_probability.png"
            plot_probability_curves(
                rows=all_rows,
                levels=levels,
                out_path=prob_path,
                min_pairs=args.plot_min_pairs,
            )
            print(f"Saved plot: {prob_path}")

    print("Done.")
    print(f"CSV: {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
