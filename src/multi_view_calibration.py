"""Validation-only view calibration and document-level score fusion."""

import math

import torch
import torch.nn.functional as F


def _fit_temperature_bias(scores, labels, max_iter):
    if not scores:
        raise ValueError("view has no validation candidates")
    if not any(labels) or all(labels):
        raise ValueError("view calibration requires both positive and negative candidates")
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    label_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(min=1e-4, max=1e4)
        logits = score_tensor / temperature + bias
        loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(min=1e-4, max=1e4))
    fitted_bias = float(bias.detach())
    if not math.isfinite(temperature) or not math.isfinite(fitted_bias):
        raise RuntimeError("non-finite calibration parameters")
    return temperature, fitted_bias


def fit_calibration_from_rows(rows, n_views: int, max_iter: int = 100):
    if not rows:
        raise ValueError("calibration rows are empty")
    if any(row.get("split_name") != "validation" for row in rows):
        raise ValueError("calibration may only be fit from validation candidates")
    scores_by_view = [[] for _ in range(n_views)]
    labels_by_view = [[] for _ in range(n_views)]
    for row in rows:
        gold = {int(doc_idx) for doc_idx in row["gold_document_ids"]}
        for candidate in row["candidates"]:
            label = int(int(candidate["document_id"]) in gold)
            for view_key, score in candidate["per_view_best_score"].items():
                view = int(view_key)
                if not 0 <= view < n_views:
                    raise ValueError(f"candidate contains unexpected view {view}")
                scores_by_view[view].append(float(score))
                labels_by_view[view].append(label)

    views = {}
    for view in range(n_views):
        temperature, bias = _fit_temperature_bias(
            scores_by_view[view], labels_by_view[view], max_iter
        )
        views[str(view)] = {
            "temperature": temperature,
            "bias": bias,
            "n_candidates": len(scores_by_view[view]),
            "n_positives": sum(labels_by_view[view]),
        }
    return {"fit_split": "validation", "n_views": n_views, "views": views}


def calibrated_fusion_score(candidate, calibration, mode: str = "lse"):
    calibrated = []
    for view_key, raw_score in candidate["per_view_best_score"].items():
        view = str(view_key)
        if view not in calibration["views"]:
            raise ValueError(f"missing calibration parameters for view {view}")
        params = calibration["views"][view]
        calibrated.append(
            float(raw_score) / float(params["temperature"])
            + float(params["bias"])
        )
    if not calibrated:
        raise ValueError("candidate has no observed view evidence")
    if mode == "sum":
        return sum(calibrated)
    if mode == "max":
        return max(calibrated)
    if mode == "lse":
        peak = max(calibrated)
        return peak + math.log(sum(math.exp(score - peak) for score in calibrated))
    raise ValueError(f"unknown fusion mode {mode!r}")


def rank_by_calibrated_fusion(candidates, calibration, mode: str = "lse"):
    def key(doc_idx):
        candidate = candidates[doc_idx]
        fusion_score = calibrated_fusion_score(candidate, calibration, mode)
        best_rank = min(candidate["per_view_rank"].values())
        return (-fusion_score, best_rank, doc_idx)

    return sorted(candidates, key=key)
