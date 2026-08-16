"""Route-level diagnostics for contiguous multi-view RQ codes."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _entropy_from_counts(counts: np.ndarray) -> float:
    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def _row_labels(rows: np.ndarray):
    _, labels, counts = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    return labels, counts


def _normalized_mutual_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    counts_a = np.bincount(labels_a)
    counts_b = np.bincount(labels_b)
    joint_pairs = np.stack([labels_a, labels_b], axis=1)
    unique_pairs, joint_counts = np.unique(joint_pairs, axis=0, return_counts=True)
    n = len(labels_a)
    pxy = joint_counts.astype(np.float64) / n
    px = counts_a[unique_pairs[:, 0]].astype(np.float64) / n
    py = counts_b[unique_pairs[:, 1]].astype(np.float64) / n
    mutual_information = float((pxy * np.log(pxy / (px * py))).sum())
    entropy_a = _entropy_from_counts(counts_a[counts_a > 0])
    entropy_b = _entropy_from_counts(counts_b[counts_b > 0])
    denominator = math.sqrt(entropy_a * entropy_b)
    return mutual_information / denominator if denominator > 0 else 0.0


def compute_code_diagnostics(codes: np.ndarray, view_size: int, n_codes: int = None):
    if codes.ndim != 2 or codes.shape[1] % view_size != 0:
        raise ValueError(
            f"codes must be 2D and divisible by view_size; got {codes.shape}, {view_size}"
        )
    n_documents, n_positions = codes.shape
    n_views = n_positions // view_size
    view_codes = codes.reshape(n_documents, n_views, view_size)

    position_metrics = []
    inferred_n_codes = int(codes.max()) + 1 if n_codes is None else int(n_codes)
    for position in range(n_positions):
        _, counts = np.unique(codes[:, position], return_counts=True)
        entropy = _entropy_from_counts(counts)
        position_metrics.append(
            {
                "position": position,
                "entropy_nats": entropy,
                "perplexity": math.exp(entropy),
                "used_codes": int(len(counts)),
                "utilization": float(len(counts) / inferred_n_codes),
            }
        )

    views = []
    view_labels = []
    collided_by_view = []
    for view in range(n_views):
        labels, counts = _row_labels(view_codes[:, view])
        route_sizes = counts[labels]
        collided = route_sizes > 1
        view_labels.append(labels)
        collided_by_view.append(collided)
        views.append(
            {
                "view": view,
                "positions": list(range(view * view_size, (view + 1) * view_size)),
                "unique_codes": int(len(counts)),
                "unique_code_ratio": float(len(counts) / n_documents),
                "collided_document_fraction": float(collided.mean()),
                "singleton_document_fraction": float((~collided).mean()),
                "max_collision_group": int(counts.max()),
                "mean_posting_list_size_per_document": float(route_sizes.mean()),
                "route_entropy_nats": _entropy_from_counts(counts),
            }
        )

    collided_matrix = np.stack(collided_by_view, axis=1)
    pairwise = []
    for first in range(n_views):
        for second in range(first + 1, n_views):
            intersection = np.logical_and(
                collided_matrix[:, first], collided_matrix[:, second]
            ).sum()
            union = np.logical_or(
                collided_matrix[:, first], collided_matrix[:, second]
            ).sum()
            pairwise.append(
                {
                    "views": [first, second],
                    "route_label_nmi": _normalized_mutual_information(
                        view_labels[first], view_labels[second]
                    ),
                    "collision_indicator_jaccard": float(intersection / union)
                    if union
                    else 0.0,
                }
            )

    return {
        "n_documents": int(n_documents),
        "n_positions": int(n_positions),
        "n_views": int(n_views),
        "view_size": int(view_size),
        "positions": position_metrics,
        "views": views,
        "pairwise_views": pairwise,
        "oracle_any_singleton_document_fraction": float(
            (~collided_matrix).any(axis=1).mean()
        ),
        "all_views_collided_document_fraction": float(
            collided_matrix.all(axis=1).mean()
        ),
    }


def _accumulate_vector_metrics(original: np.ndarray, reconstructed: np.ndarray):
    squared_error = ((original - reconstructed) ** 2).sum(axis=1)
    dot = (original * reconstructed).sum(axis=1)
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(
        reconstructed, axis=1
    )
    cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    original_energy = (original**2).sum(axis=1)
    return (
        float(squared_error.sum()),
        float(cosine.sum()),
        float(original_energy.sum()),
        len(original),
    )


def compute_reconstruction_diagnostics(
    embeddings_path: Path,
    run_dir: Path,
    codes: np.ndarray,
    view_size: int,
    quantizer_type: str,
    batch_size: int = 2048,
):
    embeddings = np.load(embeddings_path, mmap_mode="r")
    codebooks = np.load(run_dir / "rq_codebooks.npy", mmap_mode="r")
    full_recon = np.load(run_dir / "recon_embeddings.npy", mmap_mode="r")
    if len(embeddings) != len(codes):
        raise ValueError("embedding and code counts differ")

    n_views = codes.shape[1] // view_size
    view_totals = [dict(error=0.0, cosine=0.0, energy=0.0, count=0) for _ in range(n_views)]
    local_error = [0.0] * n_views
    local_energy = [0.0] * n_views
    full_totals = dict(error=0.0, cosine=0.0, energy=0.0, count=0)

    if quantizer_type == "pq":
        subspace_dims = json.load(open(run_dir / "subspace_dims.json"))
        offsets = np.cumsum([0] + subspace_dims).tolist()
    elif quantizer_type == "rq":
        subspace_dims = None
        offsets = None
    else:
        raise ValueError(f"unsupported quantizer type: {quantizer_type}")

    for start in range(0, len(codes), batch_size):
        end = min(start + batch_size, len(codes))
        original = np.asarray(embeddings[start:end], dtype=np.float32)
        batch_codes = codes[start:end]
        values = _accumulate_vector_metrics(
            original, np.asarray(full_recon[start:end], dtype=np.float32)
        )
        for key, value in zip(full_totals, values):
            full_totals[key] += value

        for view in range(n_views):
            first_level = view * view_size
            last_level = first_level + view_size
            reconstructed = np.zeros_like(original)
            if quantizer_type == "rq":
                for level in range(first_level, last_level):
                    reconstructed += codebooks[level, batch_codes[:, level]]
                local_error[view] = None
                local_energy[view] = None
            else:
                for level in range(first_level, last_level):
                    width = subspace_dims[level]
                    left, right = offsets[level : level + 2]
                    reconstructed[:, left:right] = codebooks[
                        level, batch_codes[:, level], :width
                    ]
                    delta = original[:, left:right] - reconstructed[:, left:right]
                    local_error[view] += float((delta**2).sum())
                    local_energy[view] += float((original[:, left:right] ** 2).sum())

            values = _accumulate_vector_metrics(original, reconstructed)
            for key, value in zip(view_totals[view], values):
                view_totals[view][key] += value

    def summarize(total):
        return {
            "squared_l2_per_document": total["error"] / total["count"],
            "mean_cosine": total["cosine"] / total["count"],
            "explained_energy_fraction": 1.0 - total["error"] / total["energy"],
        }

    views = []
    for view, total in enumerate(view_totals):
        summary = summarize(total)
        if quantizer_type == "pq":
            summary["covered_subspace_squared_l2_per_document"] = (
                local_error[view] / total["count"]
            )
            summary["covered_subspace_explained_energy_fraction"] = (
                1.0 - local_error[view] / local_energy[view]
            )
            summary["covered_dimensions"] = int(
                sum(subspace_dims[view * view_size : (view + 1) * view_size])
            )
        views.append(summary)

    return {"full": summarize(full_totals), "views": views}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--quantizer-type", choices=["rq", "pq"], required=True)
    parser.add_argument("--view-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    codes = np.load(args.run_dir / "rq_codes.npy", mmap_mode="r")
    codebook_size = np.load(
        args.run_dir / "rq_codebooks.npy", mmap_mode="r"
    ).shape[1]
    result = compute_code_diagnostics(codes, args.view_size, codebook_size)
    result["quantizer_type"] = args.quantizer_type
    result["reconstruction"] = compute_reconstruction_diagnostics(
        args.embeddings,
        args.run_dir,
        codes,
        args.view_size,
        args.quantizer_type,
        args.batch_size,
    )

    output = args.output or args.run_dir / "multi_view_diagnostics.json"
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
