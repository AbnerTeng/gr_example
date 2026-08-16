"""Query-side marginal and cross-view conditional predictability for RQ slices."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def summarize_view_complementarity(predictions, targets, view_size: int):
    predictions = predictions.cpu()
    targets = targets.cpu()
    n_views = predictions.shape[1] // view_size
    exact = torch.stack(
        [
            (
                predictions[:, view * view_size : (view + 1) * view_size]
                == targets[:, view * view_size : (view + 1) * view_size]
            ).all(dim=-1)
            for view in range(n_views)
        ],
        dim=1,
    )
    pairwise = []
    failures = ~exact
    for first in range(n_views):
        for second in range(first + 1, n_views):
            intersection = (failures[:, first] & failures[:, second]).sum().item()
            union = (failures[:, first] | failures[:, second]).sum().item()
            pairwise.append(
                {
                    "views": [first, second],
                    "failure_jaccard": intersection / union if union else 0.0,
                }
            )
    return {
        "any_view_exact_fraction": exact.any(dim=1).float().mean().item(),
        "all_views_wrong_fraction": failures.all(dim=1).float().mean().item(),
        "pairwise": pairwise,
    }


def summarize_views(level_metrics, predictions, targets, view_size: int):
    predictions = predictions.cpu()
    targets = targets.cpu()
    if predictions.shape != targets.shape or predictions.shape[1] % view_size != 0:
        raise ValueError("prediction and target shapes must match and divide into views")
    summaries = []
    for start in range(0, predictions.shape[1], view_size):
        stop = start + view_size
        metrics = level_metrics[start:stop]
        summaries.append(
            {
                "view": start // view_size,
                "levels": list(range(start, stop)),
                "mean_nll": sum(item["nll"] for item in metrics) / view_size,
                "mean_top1_accuracy": sum(
                    item["top1_accuracy"] for item in metrics
                )
                / view_size,
                "mean_top10_accuracy": sum(
                    item["top10_accuracy"] for item in metrics
                )
                / view_size,
                "exact_match": (
                    predictions[:, start:stop] == targets[:, start:stop]
                )
                .all(dim=-1)
                .float()
                .mean()
                .item(),
            }
        )
    return summaries


def fit_probe(
    train_query: torch.Tensor,
    train_codes: torch.Tensor,
    test_query: torch.Tensor,
    test_codes: torch.Tensor,
    target_level: int,
    conditioning,
    n_codes: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    torch.manual_seed(seed)
    input_dim = train_query.shape[-1] + len(conditioning) * n_codes
    if hidden:
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_codes),
        ).to(train_query.device)
    else:
        model = torch.nn.Linear(input_dim, n_codes).to(train_query.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_query), device=train_query.device)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            features = append_code_features(
                train_query[batch], train_codes[batch], conditioning, n_codes
            )
            logits = model(features)
            loss = F.cross_entropy(logits, train_codes[batch, target_level])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    train_nll_sum = 0.0
    train_correct = 0
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(train_query), batch_size):
            end = start + batch_size
            features = append_code_features(
                train_query[start:end], train_codes[start:end], conditioning, n_codes
            )
            logits = model(features)
            targets = train_codes[start:end, target_level]
            train_nll_sum += F.cross_entropy(
                logits, targets, reduction="sum"
            ).item()
            train_correct += (logits.argmax(dim=-1) == targets).sum().item()
        for start in range(0, len(test_query), batch_size):
            end = start + batch_size
            features = append_code_features(
                test_query[start:end], test_codes[start:end], conditioning, n_codes
            )
            all_logits.append(model(features))
    logits = torch.cat(all_logits)
    metrics, predictions = metrics_from_logits(
        logits, test_codes[:, target_level]
    )
    metrics["train_nll"] = train_nll_sum / len(train_query)
    metrics["train_top1_accuracy"] = train_correct / len(train_query)
    return metrics, predictions.cpu()


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    predictions = logits.argmax(dim=-1)
    topk = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
    metrics = {
        "nll": F.cross_entropy(logits, targets).item(),
        "top1_accuracy": (predictions == targets).float().mean().item(),
        "top10_accuracy": (topk == targets[:, None]).any(dim=-1).float().mean().item(),
    }
    return metrics, predictions


def append_code_features(
    query: torch.Tensor,
    codes: torch.Tensor,
    levels,
    n_codes: int,
):
    if not levels:
        return query
    prefix = torch.cat(
        [F.one_hot(codes[:, level], n_codes).to(query.dtype) for level in levels],
        dim=-1,
    )
    return torch.cat([query, prefix], dim=-1)


def conditioning_levels(level: int, view_size: int, cross_view: bool):
    if level < 0 or view_size <= 0:
        raise ValueError("level must be non-negative and view_size must be positive")
    view_start = (level // view_size) * view_size
    start = 0 if cross_view else view_start
    return list(range(start, level))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-query-embeddings",
        default="data/nq320k/train_query_embeddings.npy",
    )
    parser.add_argument(
        "--train-query-docidx", default="data/nq320k/train_query_docidx.npy"
    )
    parser.add_argument(
        "--test-query-embeddings",
        default="data/nq320k/test_query_embeddings.npy",
    )
    parser.add_argument(
        "--test-query-docidx", default="data/nq320k/test_query_docidx.npy"
    )
    parser.add_argument("--rq-codes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--predictions-output",
        help="optional NPZ with test targets and per-mode top-1 predictions",
    )
    parser.add_argument("--view-size", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--max-test", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    all_codes_np = np.load(args.rq_codes).astype(np.int64)
    if all_codes_np.shape[1] % args.view_size != 0:
        raise ValueError("RQ code length must divide evenly into views")
    n_codes = int(all_codes_np.max()) + 1

    train_docidx = np.load(args.train_query_docidx)
    test_docidx = np.load(args.test_query_docidx)
    train_query_np = np.load(args.train_query_embeddings, mmap_mode="r")
    test_query_np = np.load(args.test_query_embeddings, mmap_mode="r")
    if args.max_train > 0:
        train_docidx = train_docidx[: args.max_train]
        train_query_np = train_query_np[: args.max_train]
    if args.max_test > 0:
        test_docidx = test_docidx[: args.max_test]
        test_query_np = test_query_np[: args.max_test]
    if len(train_query_np) != len(train_docidx) or len(test_query_np) != len(test_docidx):
        raise ValueError("query embeddings and document-index arrays are misaligned")

    train_query = F.normalize(
        torch.from_numpy(np.asarray(train_query_np, dtype=np.float32).copy()).to(device),
        dim=-1,
    )
    test_query = F.normalize(
        torch.from_numpy(np.asarray(test_query_np, dtype=np.float32).copy()).to(device),
        dim=-1,
    )
    train_codes = torch.from_numpy(all_codes_np[train_docidx]).to(device)
    test_codes = torch.from_numpy(all_codes_np[test_docidx]).to(device)
    n_levels = train_codes.shape[1]

    result = {
        "n_train_queries": len(train_query),
        "n_test_queries": len(test_query),
        "n_levels": n_levels,
        "n_codes": n_codes,
        "view_size": args.view_size,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "seed": args.seed,
        "modes": {},
    }
    predictions_by_mode = {}
    metrics_by_mode = {}

    for mode in ["marginal", "conditional"]:
        level_metrics = []
        predictions = torch.empty(len(test_codes), n_levels, dtype=torch.long)
        for level in range(n_levels):
            if mode == "conditional" and level < args.view_size:
                level_metrics.append(dict(metrics_by_mode["marginal"][level]))
                predictions[:, level] = predictions_by_mode["marginal"][:, level]
                continue
            conditioning = conditioning_levels(
                level,
                args.view_size,
                cross_view=(mode == "conditional"),
            )
            metrics, level_predictions = fit_probe(
                train_query,
                train_codes,
                test_query,
                test_codes,
                target_level=level,
                conditioning=conditioning,
                n_codes=n_codes,
                hidden=args.hidden,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + level,
            )
            train_counts = torch.bincount(
                train_codes[:, level], minlength=n_codes
            ).float()
            probabilities = train_counts[train_counts > 0] / train_counts.sum()
            entropy = -(probabilities * probabilities.log()).sum().item()
            majority_accuracy = (
                test_codes[:, level] == train_counts.argmax()
            ).float().mean().item()
            metrics.update(
                {
                    "level": level,
                    "conditioning_levels": conditioning,
                    "target_entropy_nats": entropy,
                    "normalized_nll": metrics["nll"] / entropy if entropy else 0.0,
                    "majority_accuracy": majority_accuracy,
                }
            )
            level_metrics.append(metrics)
            predictions[:, level] = level_predictions
            print(
                f"{mode:11s} L{level} cond={conditioning} "
                f"nll={metrics['nll']:.4f} top1={metrics['top1_accuracy']:.4f} "
                f"top10={metrics['top10_accuracy']:.4f}"
            )

        metrics_by_mode[mode] = level_metrics
        predictions_by_mode[mode] = predictions
        result["modes"][mode] = {
            "levels": level_metrics,
            "views": summarize_views(
                level_metrics, predictions, test_codes, args.view_size
            ),
            "complementarity": summarize_view_complementarity(
                predictions, test_codes, args.view_size
            ),
        }

    gaps = []
    for view in range(1, n_levels // args.view_size):
        marginal = result["modes"]["marginal"]["views"][view]
        conditional = result["modes"]["conditional"]["views"][view]
        gaps.append(
            {
                "view": view,
                "marginal_minus_conditional_nll": (
                    marginal["mean_nll"] - conditional["mean_nll"]
                ),
                "conditional_minus_marginal_exact_match": (
                    conditional["exact_match"] - marginal["exact_match"]
                ),
            }
        )
    result["marginal_conditional_gaps"] = gaps

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    if args.predictions_output:
        predictions_output = Path(args.predictions_output)
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_output,
            targets=test_codes.cpu().numpy(),
            marginal=predictions_by_mode["marginal"].numpy(),
            conditional=predictions_by_mode["conditional"].numpy(),
        )
        print(f"saved predictions: {predictions_output}")
    print(json.dumps({"views": {m: result["modes"][m]["views"] for m in result["modes"]}, "gaps": gaps}, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
