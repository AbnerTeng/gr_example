import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


Pair = Tuple[int, int]
RQ_PATTERN = re.compile(r"<r\d+_(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether documents sharing longer RQ prefix tokens are more "
            "similar in embedding space than random pairs."
        )
    )
    parser.add_argument("--doc-embeddings", type=str, default="data/doc_embeddings.npy")
    parser.add_argument("--rq-codes", type=str, default="data/rq_codes.npy")
    parser.add_argument("--idx-to-rqid", type=str, default="data/idx_to_rqid.json")
    parser.add_argument("--output-dir", type=str, default="outputs/rq_semanticity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=100000,
        help="Target pair count per bucket (random/same-L*).",
    )
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=500,
        help="Initial cap per prefix group to avoid domination by huge groups.",
    )
    parser.add_argument(
        "--max-pairs-per-group-hard",
        type=int,
        default=50000,
        help="Upper cap if adaptive cap increase is required.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Maximum prefix levels to evaluate. Uses min(levels, rq_code_width).",
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


def sample_pairs_from_group(
    indices: Sequence[int], target_count: int, rng: np.random.Generator
) -> List[Pair]:
    n = len(indices)
    max_pairs = n * (n - 1) // 2
    take = min(target_count, max_pairs)
    if take <= 0:
        return []

    idx = np.asarray(indices, dtype=np.int64)
    if max_pairs <= 2000 or n <= 64:
        pairs = [(int(idx[i]), int(idx[j])) for i in range(n - 1) for j in range(i + 1, n)]
        if len(pairs) <= take:
            return pairs
        choice = rng.choice(len(pairs), size=take, replace=False)
        return [pairs[int(i)] for i in choice]

    chosen: set[Pair] = set()
    attempts, max_attempts = 0, take * 40
    while len(chosen) < take and attempts < max_attempts:
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n - 1))
        if b >= a:
            b += 1
        i, j = int(idx[a]), int(idx[b])
        if i > j:
            i, j = j, i
        chosen.add((i, j))
        attempts += 1

    if len(chosen) < take:
        perm = rng.permutation(n)
        for left_pos in range(n - 1):
            i = int(idx[perm[left_pos]])
            for right_pos in range(left_pos + 1, n):
                j = int(idx[perm[right_pos]])
                pair = (i, j) if i < j else (j, i)
                chosen.add(pair)
                if len(chosen) >= take:
                    break
            if len(chosen) >= take:
                break

    return list(chosen)


def choose_effective_group_cap(
    group_pair_caps: Sequence[int],
    target_pairs: int,
    base_cap: int,
    hard_cap: int,
) -> Tuple[int, int]:
    cap = max(1, base_cap)
    hard_cap = max(cap, hard_cap)
    while cap < hard_cap:
        total = int(sum(min(g, cap) for g in group_pair_caps))
        if total >= target_pairs:
            return cap, total
        cap = min(cap * 2, hard_cap)
    total = int(sum(min(g, cap) for g in group_pair_caps))
    return cap, total


def sample_same_prefix_pairs(
    rq_codes: np.ndarray,
    level: int,
    target_pairs: int,
    base_group_cap: int,
    hard_group_cap: int,
    rng: np.random.Generator,
) -> Tuple[List[Pair], Dict[str, int]]:
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for idx in range(rq_codes.shape[0]):
        key = tuple(int(x) for x in rq_codes[idx, :level])
        groups[key].append(idx)

    valid_groups = [g for g in groups.values() if len(g) >= 2]
    group_caps = [len(g) * (len(g) - 1) // 2 for g in valid_groups]
    if not valid_groups:
        return [], {
            "n_groups_total": len(groups),
            "n_groups_valid": 0,
            "effective_group_cap": 0,
            "candidate_pairs_before_target_sample": 0,
        }

    effective_cap, candidate_capacity = choose_effective_group_cap(
        group_caps,
        target_pairs=target_pairs,
        base_cap=base_group_cap,
        hard_cap=hard_group_cap,
    )

    candidates: List[Pair] = []
    for group in valid_groups:
        candidates.extend(sample_pairs_from_group(group, effective_cap, rng))

    if len(candidates) > target_pairs:
        choice = rng.choice(len(candidates), size=target_pairs, replace=False)
        sampled = [candidates[int(i)] for i in choice]
    else:
        sampled = candidates

    return sampled, {
        "n_groups_total": len(groups),
        "n_groups_valid": len(valid_groups),
        "effective_group_cap": int(effective_cap),
        "candidate_pairs_before_target_sample": len(candidates),
        "candidate_capacity_estimate": int(candidate_capacity),
    }


def sample_random_pairs(
    n_docs: int,
    target_pairs: int,
    rng: np.random.Generator,
    rq_codes: np.ndarray | None = None,
    require_diff_l1: bool = False,
) -> List[Pair]:
    picked: set[Pair] = set()
    attempts = 0
    max_attempts = target_pairs * 80

    while len(picked) < target_pairs and attempts < max_attempts:
        i = int(rng.integers(0, n_docs))
        j = int(rng.integers(0, n_docs - 1))
        if j >= i:
            j += 1
        if require_diff_l1 and rq_codes is not None and rq_codes[i, 0] == rq_codes[j, 0]:
            attempts += 1
            continue
        pair = (i, j) if i < j else (j, i)
        picked.add(pair)
        attempts += 1

    if len(picked) < target_pairs:
        raise RuntimeError(
            f"Failed to sample {target_pairs} unique random pairs (got {len(picked)}). "
            "Try a smaller target-pairs value."
        )
    return list(picked)


def pairs_to_cosine(embs: np.ndarray, pairs: Sequence[Pair]) -> np.ndarray:
    if not pairs:
        return np.asarray([], dtype=np.float32)
    left = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    right = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    return (embs[left] * embs[right]).sum(axis=1).astype(np.float32)


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n_pairs": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    n = values.size
    mean = float(values.mean())
    median = float(np.median(values))
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    q25, q75 = np.percentile(values, [25, 75]).tolist()
    sem = std / math.sqrt(n) if n > 1 else 0.0
    ci95_low = mean - 1.96 * sem
    ci95_high = mean + 1.96 * sem
    return {
        "n_pairs": int(n),
        "mean": mean,
        "median": median,
        "std": std,
        "q25": float(q25),
        "q75": float(q75),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
    }


def mann_whitney_and_effects(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if x.size == 0 or y.size == 0:
        raise ValueError("Both samples must be non-empty for significance testing.")

    n1, n2 = int(x.size), int(y.size)
    combined = np.concatenate([x, y]).astype(np.float64)
    order = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[order]

    sorted_ranks = np.empty(sorted_vals.shape[0], dtype=np.float64)
    tie_term = 0.0
    i, total_n = 0, sorted_vals.shape[0]
    while i < total_n:
        j = i + 1
        while j < total_n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        sorted_ranks[i:j] = avg_rank
        t = j - i
        if t > 1:
            tie_term += t**3 - t
        i = j

    inv = np.empty_like(order)
    inv[order] = np.arange(total_n)
    ranks = sorted_ranks[inv]

    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_min = min(u1, u2)
    u_mean = n1 * n2 / 2.0

    tie_corr = 1.0
    if total_n > 1:
        tie_corr = 1.0 - tie_term / (total_n**3 - total_n)
    var_u = (n1 * n2 * (total_n + 1) / 12.0) * tie_corr

    if var_u <= 0:
        z = 0.0
        p_two_sided = 1.0
    else:
        z = (abs(u1 - u_mean) - 0.5) / math.sqrt(var_u)
        p_two_sided = math.erfc(abs(z) / math.sqrt(2.0))

    cliffs_delta = (2.0 * u1) / (n1 * n2) - 1.0

    mean_diff = float(x.mean() - y.mean())
    x_var = float(x.var(ddof=1)) if n1 > 1 else 0.0
    y_var = float(y.var(ddof=1)) if n2 > 1 else 0.0
    pooled_var = ((n1 - 1) * x_var + (n2 - 1) * y_var) / max(1, (n1 + n2 - 2))
    cohen_d = mean_diff / math.sqrt(pooled_var) if pooled_var > 0 else float("nan")

    return {
        "n_x": n1,
        "n_y": n2,
        "u1": float(u1),
        "u2": float(u2),
        "u_min": float(u_min),
        "z_approx": float(z),
        "p_two_sided_approx": float(p_two_sided),
        "cliffs_delta": float(cliffs_delta),
        "cohens_d": float(cohen_d),
        "mean_diff": float(mean_diff),
    }


def shared_prefix_ratio(pairs: Sequence[Pair], rq_codes: np.ndarray, level: int) -> float:
    if not pairs:
        return float("nan")
    left = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    right = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    shared = np.all(rq_codes[left, :level] == rq_codes[right, :level], axis=1)
    return float(shared.mean())


def write_summary_csv(path: Path, rows: Dict[str, Dict[str, float]]) -> None:
    fieldnames = [
        "pair_type",
        "n_pairs",
        "mean",
        "median",
        "std",
        "q25",
        "q75",
        "ci95_low",
        "ci95_high",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, stats in rows.items():
            row = {"pair_type": key}
            row.update(stats)
            writer.writerow(row)


def write_comparison_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = [
        "left",
        "right",
        "n_x",
        "n_y",
        "u1",
        "u2",
        "u_min",
        "z_approx",
        "p_two_sided_approx",
        "cliffs_delta",
        "cohens_d",
        "mean_diff",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_boxplot(distributions: Dict[str, np.ndarray], out_path: Path) -> None:
    labels = list(distributions.keys())
    values = [distributions[k] for k in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity Distribution by Pair Type")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prefix_curve(
    summaries: Dict[str, Dict[str, float]],
    n_levels: int,
    out_path: Path,
) -> None:
    baseline = "random_all"

    x_vals = [0]
    y_vals = [summaries[baseline]["mean"]]
    labels = [baseline]
    for level in range(1, n_levels + 1):
        key = f"same_l{level}"
        if key in summaries:
            x_vals.append(level)
            y_vals.append(summaries[key]["mean"])
            labels.append(key)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_vals, marker="o")
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xlabel("Shared Prefix Length")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Prefix Length vs Mean Similarity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cdf(distributions: Dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values in distributions.items():
        if values.size == 0:
            continue
        x = np.sort(values)
        y = np.arange(1, values.size + 1) / values.size
        ax.plot(x, y, label=label, linewidth=1.2)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Pair Similarities")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
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
    n_levels = min(args.levels, rq_codes.shape[1])
    print(f"  rq_codes: {rq_codes.shape}, evaluating levels=1..{n_levels}")

    pair_sets: Dict[str, List[Pair]] = {}
    pair_meta: Dict[str, Dict[str, float]] = {}

    print("Sampling random-all pairs...")
    pair_sets["random_all"] = sample_random_pairs(
        n_docs=embs.shape[0],
        target_pairs=args.target_pairs,
        rng=rng,
        rq_codes=rq_codes,
        require_diff_l1=False,
    )
    pair_meta["random_all"] = {
        "shared_l1_ratio": shared_prefix_ratio(pair_sets["random_all"], rq_codes, 1)
    }

    for level in range(1, n_levels + 1):
        key = f"same_l{level}"
        print(f"Sampling {key} pairs...")
        pairs, meta = sample_same_prefix_pairs(
            rq_codes=rq_codes,
            level=level,
            target_pairs=args.target_pairs,
            base_group_cap=args.max_pairs_per_group,
            hard_group_cap=args.max_pairs_per_group_hard,
            rng=rng,
        )
        pair_sets[key] = pairs
        pair_meta[key] = meta
        pair_meta[key]["shared_l1_ratio"] = shared_prefix_ratio(pairs, rq_codes, 1)
        print(
            f"  {key}: sampled={len(pairs)}, valid_groups={meta.get('n_groups_valid', 0)}, "
            f"cap={meta.get('effective_group_cap', 0)}"
        )

    print("Computing cosine similarities...")
    distributions: Dict[str, np.ndarray] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    for key, pairs in pair_sets.items():
        sims = pairs_to_cosine(embs, pairs)
        distributions[key] = sims
        summaries[key] = summarize_distribution(sims)

    comparisons: List[Dict[str, float]] = []
    if "same_l1" in distributions:
        stats = mann_whitney_and_effects(distributions["same_l1"], distributions["random_all"])
        stats.update({"left": "same_l1", "right": "random_all"})
        comparisons.append(stats)

    for level in range(2, n_levels + 1):
        left = f"same_l{level}"
        right = f"same_l{level-1}"
        if left in distributions and right in distributions:
            stats = mann_whitney_and_effects(distributions[left], distributions[right])
            stats.update({"left": left, "right": right})
            comparisons.append(stats)

    summary_csv = output_dir / "summary_stats.csv"
    comparison_csv = output_dir / "significance_tests.csv"
    summary_json = output_dir / "results.json"
    write_summary_csv(summary_csv, summaries)
    write_comparison_csv(comparison_csv, comparisons)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "embeddings_shape": list(embs.shape),
        "rq_codes_shape": list(rq_codes.shape),
        "evaluated_levels": n_levels,
        "pair_meta": pair_meta,
        "summary_stats": summaries,
        "significance_tests": comparisons,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.no_plots:
        print("Skipping plots (--no-plots).")
    else:
        if plt is None:
            print("matplotlib not installed, skipping plot generation.")
        else:
            boxplot_path = output_dir / "boxplot.png"
            prefix_curve_path = output_dir / "prefix_curve.png"
            cdf_path = output_dir / "cdf.png"
            ordered_keys = ["random_all"]
            for level in range(1, n_levels + 1):
                key = f"same_l{level}"
                if key in distributions:
                    ordered_keys.append(key)
            ordered_distributions = {k: distributions[k] for k in ordered_keys}

            plot_boxplot(ordered_distributions, boxplot_path)
            plot_prefix_curve(summaries, n_levels, prefix_curve_path)
            plot_cdf(ordered_distributions, cdf_path)
            print(f"Saved plots: {boxplot_path}, {prefix_curve_path}, {cdf_path}")

    print("Done.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Significance CSV: {comparison_csv}")
    print(f"JSON report: {summary_json}")


if __name__ == "__main__":
    main()
