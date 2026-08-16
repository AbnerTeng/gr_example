"""Lightweight document reranking for saved Multi-DocID candidates."""

import hashlib
import math

import numpy as np


def stable_is_dev(query_index: int, fraction: float = 0.2, seed: int = 42) -> bool:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    digest = hashlib.sha256(f"{seed}:{int(query_index)}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < fraction


def feature_names(n_views: int):
    names = []
    for view in range(n_views):
        names.extend(
            [
                f"view{view}_score",
                f"view{view}_present",
                f"view{view}_reciprocal_rank",
                f"view{view}_log_collision",
            ]
        )
    return names + ["hit_count", "best_reciprocal_rank", "calibrated_lse", "calibrated_max", "calibrated_mean"]


def _get(mapping, view):
    return mapping.get(str(view), mapping.get(view))


def extract_features(candidate, calibration, score_floors, n_views: int):
    observed_scores = []
    features = []
    for view in range(n_views):
        raw_score = _get(candidate["per_view_best_score"], view)
        rank = _get(candidate["per_view_rank"], view)
        collision = _get(candidate["collision_size"], view)
        present = raw_score is not None
        if present:
            params = calibration["views"][str(view)]
            score = float(raw_score) / float(params["temperature"]) + float(params["bias"])
            observed_scores.append(score)
            reciprocal_rank = 1.0 / float(rank)
            log_collision = math.log1p(float(collision))
        else:
            score = float(score_floors[view])
            reciprocal_rank = 0.0
            log_collision = 0.0
        features.extend([score, float(present), reciprocal_rank, log_collision])
    if not observed_scores:
        raise ValueError("candidate has no observed view evidence")
    peak = max(observed_scores)
    lse = peak + math.log(sum(math.exp(score - peak) for score in observed_scores))
    best_reciprocal_rank = max(
        features[4 * view + 2] for view in range(n_views)
    )
    features.extend(
        [
            float(candidate["hit_count"]),
            best_reciprocal_rank,
            lse,
            max(observed_scores),
            sum(observed_scores) / len(observed_scores),
        ]
    )
    return np.asarray(features, dtype=np.float64)


def score_floors_from_rows(rows, calibration, n_views: int):
    floors = [math.inf] * n_views
    for row in rows:
        for candidate in row["candidates"]:
            for view in range(n_views):
                raw_score = _get(candidate["per_view_best_score"], view)
                if raw_score is None:
                    continue
                params = calibration["views"][str(view)]
                score = float(raw_score) / float(params["temperature"]) + float(
                    params["bias"]
                )
                floors[view] = min(floors[view], score)
    if any(not math.isfinite(floor) for floor in floors):
        raise ValueError("every view needs observed validation scores")
    return floors


def select_training_candidates(
    row, calibration, score_floors, n_views: int, n_negatives: int
):
    gold = {int(doc_idx) for doc_idx in row["gold_document_ids"]}
    positives = []
    negatives = []
    for candidate in row["candidates"]:
        doc_idx = int(candidate["document_id"])
        if doc_idx in gold:
            positives.append(candidate)
            continue
        features = extract_features(
            candidate, calibration, score_floors, n_views
        )
        lse = float(features[-3])
        best_rank = min(int(rank) for rank in candidate["per_view_rank"].values())
        negatives.append(
            (
                -int(candidate["hit_count"]),
                -lse,
                best_rank,
                doc_idx,
                candidate,
            )
        )
    negatives.sort(key=lambda item: item[:4])
    selected = positives + [item[-1] for item in negatives[:n_negatives]]
    labels = [1] * len(positives) + [0] * min(len(negatives), n_negatives)
    return selected, labels


def fit_linear_reranker(
    rows,
    calibration,
    score_floors,
    n_views: int,
    n_negatives: int,
    regularization_c: float,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feature_blocks = []
    label_blocks = []
    n_training_queries = 0
    for row in rows:
        selected, labels = select_training_candidates(
            row,
            calibration,
            score_floors,
            n_views,
            n_negatives,
        )
        if not labels or not any(labels):
            continue
        feature_blocks.append(
            np.stack(
                [
                    extract_features(
                        candidate, calibration, score_floors, n_views
                    )
                    for candidate in selected
                ]
            )
        )
        label_blocks.append(np.asarray(labels, dtype=np.int64))
        n_training_queries += 1
    if not feature_blocks:
        raise ValueError("no training query has a positive candidate")
    x = np.concatenate(feature_blocks, axis=0)
    y = np.concatenate(label_blocks, axis=0)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    model = LogisticRegression(
        C=float(regularization_c),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    ).fit(x_scaled, y)
    scaled_coefficients = model.coef_[0]
    coefficients = scaled_coefficients / scaler.scale_
    intercept = float(
        model.intercept_[0]
        - np.sum(scaled_coefficients * scaler.mean_ / scaler.scale_)
    )
    return {
        "model_type": "linear_logistic_document_reranker",
        "n_views": int(n_views),
        "feature_names": feature_names(n_views),
        "score_floors": [float(value) for value in score_floors],
        "coefficients": coefficients.astype(float).tolist(),
        "intercept": intercept,
        "calibration": calibration,
        "training": {
            "n_queries_with_positive": n_training_queries,
            "n_examples": int(len(y)),
            "n_positives": int(y.sum()),
            "n_negatives": int((1 - y).sum()),
            "hard_negatives_per_query": int(n_negatives),
            "regularization_c": float(regularization_c),
            "class_weight": "balanced",
        },
    }


def rank_candidates(candidates, artifact):
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    expected_names = feature_names(int(artifact["n_views"]))
    if artifact["feature_names"] != expected_names:
        raise ValueError("reranker feature schema mismatch")
    ranked = []
    for doc_idx, candidate in candidates.items():
        features = extract_features(
            candidate,
            artifact["calibration"],
            artifact["score_floors"],
            int(artifact["n_views"]),
        )
        score = float(features @ coefficients + float(artifact["intercept"]))
        best_rank = min(int(rank) for rank in candidate["per_view_rank"].values())
        ranked.append((doc_idx, score, best_rank))
    ranked.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [doc_idx for doc_idx, _, _ in ranked]


def evaluate_rows(rows, artifact):
    from collections import defaultdict

    from .eval_multi_view_gr import ranking_metrics

    totals = defaultdict(float)
    n_queries = 0
    for row in rows:
        candidates = {
            int(candidate["document_id"]): candidate
            for candidate in row["candidates"]
        }
        ranked = rank_candidates(candidates, artifact)
        gold = {int(doc_idx) for doc_idx in row["gold_document_ids"]}
        metrics = ranking_metrics(ranked, gold)
        for key, value in metrics.items():
            totals[key] += value
        n_queries += 1
    if not n_queries:
        raise ValueError("evaluation rows are empty")
    return {"n_queries": n_queries, **{key: value / n_queries for key, value in totals.items()}}
