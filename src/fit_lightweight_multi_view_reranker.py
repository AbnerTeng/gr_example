"""Fit and select a lightweight Multi-DocID document reranker on validation only."""

import argparse
import hashlib
import json
from pathlib import Path

from .lightweight_multi_view_reranker import (
    evaluate_rows,
    fit_linear_reranker,
    score_floors_from_rows,
    stable_is_dev,
)
from .multi_view_calibration import fit_calibration_from_rows


def parse_csv(text, cast):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-candidates", required=True)
    parser.add_argument("--full-calibration", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--negative-counts", default="20,50")
    parser.add_argument("--regularization-c", default="0.1,1.0,10.0")
    parser.add_argument("--calibration-max-iter", type=int, default=100)
    args = parser.parse_args()

    candidate_path = Path(args.validation_candidates)
    with open(candidate_path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("split_name") != "validation" for row in rows):
        raise ValueError("reranker fitting requires validation candidates only")

    train_rows, dev_rows = [], []
    for row in rows:
        target = dev_rows if stable_is_dev(
            row["query_index"], args.dev_fraction, args.split_seed
        ) else train_rows
        target.append(row)
    if not train_rows or not dev_rows:
        raise ValueError("internal validation split is empty")

    train_calibration = fit_calibration_from_rows(
        train_rows, n_views=args.n_views, max_iter=args.calibration_max_iter
    )
    train_floors = score_floors_from_rows(
        train_rows, train_calibration, args.n_views
    )

    trials = []
    best = None
    for n_negatives in parse_csv(args.negative_counts, int):
        for regularization_c in parse_csv(args.regularization_c, float):
            artifact = fit_linear_reranker(
                train_rows,
                train_calibration,
                train_floors,
                args.n_views,
                n_negatives,
                regularization_c,
            )
            metrics = evaluate_rows(dev_rows, artifact)
            trial = {
                "hard_negatives_per_query": n_negatives,
                "regularization_c": regularization_c,
                "dev_metrics": metrics,
            }
            trials.append(trial)
            key = (
                metrics["mrr@10_doc"],
                metrics["ndcg@10_doc"],
                metrics["recall@10_doc"],
            )
            if best is None or key > best[0]:
                best = (key, n_negatives, regularization_c)

    _, selected_negatives, selected_c = best
    full_calibration = json.load(open(args.full_calibration))
    if full_calibration.get("fit_split") != "validation":
        raise ValueError("full calibration must be validation-fitted")
    full_floors = score_floors_from_rows(rows, full_calibration, args.n_views)
    final_artifact = fit_linear_reranker(
        rows,
        full_calibration,
        full_floors,
        args.n_views,
        selected_negatives,
        selected_c,
    )
    final_artifact["protocol"] = {
        "fit_split": "validation",
        "internal_split": "stable_query_hash",
        "dev_fraction": args.dev_fraction,
        "split_seed": args.split_seed,
        "n_internal_train_queries": len(train_rows),
        "n_internal_dev_queries": len(dev_rows),
        "selection_objective": [
            "mrr@10_doc",
            "ndcg@10_doc",
            "recall@10_doc",
        ],
        "selected_hard_negatives_per_query": selected_negatives,
        "selected_regularization_c": selected_c,
        "validation_candidates": str(candidate_path.resolve()),
        "validation_candidates_sha256": file_sha256(candidate_path),
        "full_calibration": str(Path(args.full_calibration).resolve()),
        "full_calibration_sha256": file_sha256(args.full_calibration),
    }
    final_artifact["full_validation_metrics"] = evaluate_rows(rows, final_artifact)

    selection = {
        "protocol": final_artifact["protocol"],
        "trials": trials,
        "selected": {
            "hard_negatives_per_query": selected_negatives,
            "regularization_c": selected_c,
        },
        "full_validation_metrics": final_artifact["full_validation_metrics"],
    }
    for path, payload in [
        (Path(args.model_output), final_artifact),
        (Path(args.selection_output), selection),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
