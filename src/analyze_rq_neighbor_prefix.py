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
            "Measure whether embedding nearest neighbors share RQ prefix tokens "
            "(PrefixRecall@K / PrefixHit@K / UniqueL1@K)."
        )
    )
    parser.add_argument("--doc-embeddings", type=str, default="data/doc_embeddings.npy")
    parser.add_argument("--rq-codes", type=str, default="data/rq_codes.npy")
    parser.add_argument("--idx-to-rqid", type=str, default="data/idx_to_rqid.json")
    parser.add_argument("--output-dir", type=str, default="outputs/rq_neighbor_prefix")
    parser.add_argument("--seed", type=int, default=42)
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
        "--anchor-docs",
        type=int,
        default=-1,
        help="Number of anchor docs. -1 means all docs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size for FAISS neighbor search.",
    )
    parser.add_argument(
        "--search-extra",
        type=int,
        default=32,
        help="Search K+extra neighbors before self-exclusion to ensure enough results.",
    )
    parser.add_argument(
        "--n-random-baselines",
        type=int,
        default=1,
        help="Number of shuffled-RQ random baselines to run.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_k_values(text: str) -> List[int]:
    ks = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"K values must be positive. Got {value}")
        ks.append(value)
    if not ks:
        raise ValueError("At least one K value is required.")
    return sorted(set(ks))


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


def select_anchor_indices(
    n_docs: int, anchor_docs: int, rng: np.random.Generator
) -> np.ndarray:
    if anchor_docs <= 0 or anchor_docs >= n_docs:
        return np.arange(n_docs, dtype=np.int64)
    return np.asarray(rng.choice(n_docs, size=anchor_docs, replace=False), dtype=np.int64)


def search_topk_neighbors(
    embs: np.ndarray,
    anchor_idx: np.ndarray,
    k_max: int,
    batch_size: int,
    search_extra: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if faiss is None:
        raise ImportError("faiss is required for nearest-neighbor search.")

    n_docs, dim = embs.shape
    raw_k = min(n_docs, k_max + max(1, search_extra) + 1)

    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    neighbors = np.empty((len(anchor_idx), k_max), dtype=np.int32)
    self_hit_rows = 0
    rows_refetched = 0

    for start in range(0, len(anchor_idx), batch_size):
        end = min(start + batch_size, len(anchor_idx))
        anchors_batch = anchor_idx[start:end]
        _, idx_batch = index.search(embs[anchors_batch].astype(np.float32), raw_k)

        for row, src in enumerate(anchors_batch):
            candidates = idx_batch[row]
            if np.any(candidates == src):
                self_hit_rows += 1
            filtered = candidates[candidates != src]

            if filtered.size < k_max:
                rows_refetched += 1
                full_k = min(n_docs, max(k_max + 1, n_docs))
                _, idx_refetch = index.search(
                    embs[src : src + 1].astype(np.float32), full_k
                )
                filtered = idx_refetch[0][idx_refetch[0] != src]

            if filtered.size < k_max:
                raise RuntimeError(
                    f"Anchor {src} has only {filtered.size} neighbors after self exclusion; "
                    f"cannot satisfy K={k_max}."
                )

            neighbors[start + row] = filtered[:k_max]

    meta = {
        "n_docs": int(n_docs),
        "n_anchors": int(len(anchor_idx)),
        "k_max": int(k_max),
        "search_k_raw": int(raw_k),
        "self_hit_rows": int(self_hit_rows),
        "rows_refetched": int(rows_refetched),
        "index_type": "IndexFlatIP",
        "approximate_search": 0,
    }
    return neighbors, meta


def empty_level_dict(levels: int) -> Dict[str, float]:
    return {f"l{level}": 0.0 for level in range(1, levels + 1)}


def compute_prefix_metrics(
    rq_codes: np.ndarray,
    anchor_idx: np.ndarray,
    neighbors: np.ndarray,
    k_values: Sequence[int],
    levels: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    anchor_codes = rq_codes[anchor_idx, :levels]
    k_max = neighbors.shape[1]

    recall_by_k = {k: empty_level_dict(levels) for k in k_values}
    hit_by_k = {k: empty_level_dict(levels) for k in k_values}

    for level in range(1, levels + 1):
        neighbor_codes = rq_codes[neighbors[:, :k_max], :level]
        match = np.all(neighbor_codes == anchor_codes[:, None, :level], axis=2)
        for k in k_values:
            sub = match[:, :k]
            recall_by_k[k][f"l{level}"] = float(sub.mean())
            hit_by_k[k][f"l{level}"] = float(sub.any(axis=1).mean())

    recall_rows = []
    hit_rows = []
    for k in k_values:
        rec_row = {"k": int(k)}
        hit_row = {"k": int(k)}
        rec_row.update(recall_by_k[k])
        hit_row.update(hit_by_k[k])
        recall_rows.append(rec_row)
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
    n_anchors = neighbors.shape[0]
    neighbor_l1 = rq_codes[neighbors, 0]
    rows = []
    distributions: Dict[int, np.ndarray] = {}

    for k in k_values:
        slice_l1 = neighbor_l1[:, :k]
        if k == 1:
            counts = np.ones(n_anchors, dtype=np.int16)
        else:
            sorted_vals = np.sort(slice_l1, axis=1)
            counts = 1 + np.sum(sorted_vals[:, 1:] != sorted_vals[:, :-1], axis=1)
        stats = summarize_unique(counts.astype(np.float32))
        row = {"k": int(k)}
        row.update(stats)
        rows.append(row)
        distributions[k] = counts

    return rows, distributions


def with_method(rows: List[Dict[str, float]], method: str) -> List[Dict[str, float]]:
    out = []
    for row in rows:
        merged = {"method": method}
        merged.update(row)
        out.append(merged)
    return out


def evaluate_method(
    method_name: str,
    rq_codes: np.ndarray,
    anchor_idx: np.ndarray,
    neighbors: np.ndarray,
    k_values: Sequence[int],
    levels: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]], Dict[int, np.ndarray]]:
    recall_rows, hit_rows = compute_prefix_metrics(
        rq_codes=rq_codes,
        anchor_idx=anchor_idx,
        neighbors=neighbors,
        k_values=k_values,
        levels=levels,
    )
    unique_rows, unique_dist = compute_unique_l1_metrics(
        rq_codes=rq_codes,
        neighbors=neighbors,
        k_values=k_values,
    )
    return (
        with_method(recall_rows, method_name),
        with_method(hit_rows, method_name),
        with_method(unique_rows, method_name),
        unique_dist,
    )


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_prefix_recall_vanilla(
    recall_rows: List[Dict[str, float]], levels: int, out_path: Path
) -> None:
    level_labels = [f"L{level}" for level in range(1, levels + 1)]
    x = np.arange(1, levels + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in recall_rows:
        y = [row[f"l{level}"] for level in range(1, levels + 1)]
        ax.plot(x, y, marker="o", label=f"K={int(row['k'])}")
    ax.set_xticks(x, level_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Prefix Level")
    ax.set_ylabel("PrefixRecall@K")
    ax.set_title("Vanilla RQ PrefixRecall@K by Level")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_unique_l1_boxplot(
    unique_distributions: Dict[int, np.ndarray], out_path: Path
) -> None:
    ks = sorted(unique_distributions.keys())
    values = [unique_distributions[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, labels=[f"K={k}" for k in ks], showfliers=False)
    ax.set_ylabel("Unique L1 Clusters in Top-K")
    ax.set_title("Vanilla RQ kNN Fragmentation (UniqueL1@K)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_l1_compare(
    recall_rows: List[Dict[str, float]], out_path: Path
) -> None:
    grouped: Dict[str, List[Tuple[int, float]]] = {}
    for row in recall_rows:
        method = str(row["method"])
        grouped.setdefault(method, []).append((int(row["k"]), float(row["l1"])))

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, points in grouped.items():
        points = sorted(points, key=lambda x: x[0])
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=method,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("L1 PrefixRecall@K")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("L1 PrefixRecall@K: Vanilla vs Random Code")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_random_baselines <= 0:
        raise ValueError("--n-random-baselines must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    k_values = parse_k_values(args.k_values)
    k_max = max(k_values)
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

    anchor_idx = select_anchor_indices(embs.shape[0], args.anchor_docs, rng)
    print(f"Using {len(anchor_idx)} anchors")

    print(f"Searching top-{k_max} embedding neighbors (exact FAISS IP)...")
    neighbors, neighbor_meta = search_topk_neighbors(
        embs=embs,
        anchor_idx=anchor_idx,
        k_max=k_max,
        batch_size=args.batch_size,
        search_extra=args.search_extra,
    )
    print(
        f"  done: neighbors shape={neighbors.shape}, "
        f"self_hit_rows={neighbor_meta['self_hit_rows']}, "
        f"rows_refetched={neighbor_meta['rows_refetched']}"
    )

    all_recall_rows: List[Dict[str, float]] = []
    all_hit_rows: List[Dict[str, float]] = []
    all_unique_rows: List[Dict[str, float]] = []

    print("Evaluating vanilla RQ...")
    vanilla_recall, vanilla_hit, vanilla_unique, vanilla_unique_dist = evaluate_method(
        method_name="vanilla_rq",
        rq_codes=rq_codes,
        anchor_idx=anchor_idx,
        neighbors=neighbors,
        k_values=k_values,
        levels=levels,
    )
    all_recall_rows.extend(vanilla_recall)
    all_hit_rows.extend(vanilla_hit)
    all_unique_rows.extend(vanilla_unique)

    random_runs_meta = []
    random_recall_runs: List[List[Dict[str, float]]] = []
    random_hit_runs: List[List[Dict[str, float]]] = []
    random_unique_runs: List[List[Dict[str, float]]] = []

    for run in range(args.n_random_baselines):
        print(f"Evaluating random-code baseline ({run + 1}/{args.n_random_baselines})...")
        perm = rng.permutation(rq_codes.shape[0])
        rq_random = rq_codes[perm]
        run_recall, run_hit, run_unique, _ = evaluate_method(
            method_name=f"random_code_{run + 1}",
            rq_codes=rq_random,
            anchor_idx=anchor_idx,
            neighbors=neighbors,
            k_values=k_values,
            levels=levels,
        )
        random_runs_meta.append({"run": run + 1, "perm_seed_state": "implicit"})
        random_recall_runs.append(run_recall)
        random_hit_runs.append(run_hit)
        random_unique_runs.append(run_unique)

    if args.n_random_baselines == 1:
        single_recall = [dict(r, method="random_code") for r in random_recall_runs[0]]
        single_hit = [dict(r, method="random_code") for r in random_hit_runs[0]]
        single_unique = [dict(r, method="random_code") for r in random_unique_runs[0]]
        all_recall_rows.extend(single_recall)
        all_hit_rows.extend(single_hit)
        all_unique_rows.extend(single_unique)
    else:
        # Keep per-run rows
        for rows in random_recall_runs:
            all_recall_rows.extend(rows)
        for rows in random_hit_runs:
            all_hit_rows.extend(rows)
        for rows in random_unique_runs:
            all_unique_rows.extend(rows)

        # Also append averaged random baseline
        for k in k_values:
            avg_recall_row = {"method": "random_code_mean", "k": k}
            avg_hit_row = {"method": "random_code_mean", "k": k}
            avg_unique_row = {"method": "random_code_mean", "k": k}
            for level in range(1, levels + 1):
                key = f"l{level}"
                avg_recall_row[key] = float(
                    np.mean([rows[k_values.index(k)][key] for rows in random_recall_runs])
                )
                avg_hit_row[key] = float(
                    np.mean([rows[k_values.index(k)][key] for rows in random_hit_runs])
                )
            for key in ["mean", "median", "q25", "q75"]:
                avg_unique_row[key] = float(
                    np.mean([rows[k_values.index(k)][key] for rows in random_unique_runs])
                )
            all_recall_rows.append(avg_recall_row)
            all_hit_rows.append(avg_hit_row)
            all_unique_rows.append(avg_unique_row)

    level_headers = [f"l{level}" for level in range(1, levels + 1)]
    recall_csv = output_dir / "prefix_recall_at_k.csv"
    hit_csv = output_dir / "prefix_hit_at_k.csv"
    unique_csv = output_dir / "unique_l1_at_k.csv"

    write_csv(recall_csv, all_recall_rows, ["method", "k", *level_headers])
    write_csv(hit_csv, all_hit_rows, ["method", "k", *level_headers])
    write_csv(unique_csv, all_unique_rows, ["method", "k", "mean", "median", "q25", "q75"])

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "k_values": k_values,
        "embeddings_shape": list(embs.shape),
        "rq_codes_shape": list(rq_codes.shape),
        "evaluated_levels": levels,
        "neighbor_search": neighbor_meta,
        "random_baseline_runs": random_runs_meta,
        "prefix_recall_at_k": all_recall_rows,
        "prefix_hit_at_k": all_hit_rows,
        "unique_l1_at_k": all_unique_rows,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.no_plots:
        print("Skipping plots (--no-plots).")
    else:
        if plt is None:
            print("matplotlib not installed, skipping plot generation.")
        else:
            vanilla_recall_rows = [r for r in all_recall_rows if r["method"] == "vanilla_rq"]
            if vanilla_recall_rows:
                plot_prefix_recall_vanilla(
                    recall_rows=sorted(vanilla_recall_rows, key=lambda r: int(r["k"])),
                    levels=levels,
                    out_path=output_dir / "prefix_recall_vanilla.png",
                )
                plot_l1_compare(
                    recall_rows=[
                        r
                        for r in all_recall_rows
                        if r["method"] in {"vanilla_rq", "random_code", "random_code_mean"}
                    ],
                    out_path=output_dir / "l1_recall_compare.png",
                )
                plot_unique_l1_boxplot(
                    unique_distributions=vanilla_unique_dist,
                    out_path=output_dir / "unique_l1_vanilla_boxplot.png",
                )

    print("Done.")
    print(f"PrefixRecall CSV: {recall_csv}")
    print(f"PrefixHit CSV: {hit_csv}")
    print(f"UniqueL1 CSV: {unique_csv}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
