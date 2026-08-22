"""Equal-per-view constrained decoding for one shared Multi-DocID T5."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    LogitsProcessorList,
    T5ForConditionalGeneration,
)

from .fast_trie import FastRQTrie, FastRQTrieLogitsProcessor
from .multi_view_calibration import rank_by_calibrated_fusion
from .multi_view_fusion import (
    aggregate_document_candidates,
    build_view_posting_lists,
    compute_view_route_log_normalizers,
    rank_by_beam_posterior,
    rank_by_majority,
)
from .multi_view_tokenizer import route_token_ids, validate_atomic_tokens
from .prep_multi_view_gr import validate_view_mapping


def build_view_tries(idx_to_view_ids, tokenizer):
    if not idx_to_view_ids:
        raise ValueError("view mapping is empty")
    n_views = len(idx_to_view_ids[0])
    routes_by_view = [[] for _ in range(n_views)]
    for doc_idx, routes in enumerate(idx_to_view_ids):
        if len(routes) != n_views:
            raise ValueError(
                f"document {doc_idx} has {len(routes)} views; expected {n_views}"
            )
        for view, route in enumerate(routes):
            routes_by_view[view].append(route)
    return [
        FastRQTrie(routes, tokenizer, tokenizer.eos_token_id)
        for routes in routes_by_view
    ]


def build_route_id_maps(idx_to_view_ids, tokenizer):
    n_views = len(idx_to_view_ids[0])
    maps = [dict() for _ in range(n_views)]
    for routes in idx_to_view_ids:
        for view, route in enumerate(routes):
            key = tuple(route_token_ids(tokenizer, route, add_eos=False))
            previous = maps[view].setdefault(key, route)
            if previous != route:
                raise ValueError("distinct route strings map to identical token IDs")
    return maps


def generated_sequence_to_route(sequence, route_id_map, eos_token_id):
    token_ids = sequence.tolist()
    if token_ids:
        token_ids = token_ids[1:]  # T5 decoder-start token
    if eos_token_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_token_id)]
    return route_id_map.get(tuple(token_ids))


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gold_documents(row):
    if "gt_doc_indices" in row:
        return {int(doc_idx) for doc_idx in row["gt_doc_indices"]}
    return {int(row["gt_doc_idx"])}


def ranking_metrics(ranked_documents, gold_documents, cutoffs=(1, 10, 100)):
    gold = set(gold_documents)
    if not gold:
        raise ValueError("query has no relevant documents")
    result = {}
    relevant_ranks = [
        rank
        for rank, doc_idx in enumerate(ranked_documents, 1)
        if doc_idx in gold
    ]
    first_rank = relevant_ranks[0] if relevant_ranks else 0
    result["mrr@10_doc"] = (
        1.0 / first_rank if 0 < first_rank <= 10 else 0.0
    )
    for cutoff in cutoffs:
        retrieved = sum(rank <= cutoff for rank in relevant_ranks)
        result[f"hit@{cutoff}_doc"] = float(retrieved > 0)
        result[f"recall@{cutoff}_doc"] = retrieved / len(gold)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in relevant_ranks
            if rank <= cutoff
        )
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(gold), cutoff) + 1)
        )
        result[f"ndcg@{cutoff}_doc"] = dcg / ideal
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-queries", required=True)
    parser.add_argument("--idx-to-view-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-output")
    parser.add_argument(
        "--split-name",
        choices=["train", "validation", "test", "smoke"],
        default="test",
    )
    parser.add_argument("--calibration")
    parser.add_argument(
        "--fusion-mode", choices=["sum", "max", "lse"], default="lse"
    )
    parser.add_argument("--beams", type=int, default=100)
    parser.add_argument(
        "--single-view",
        type=int,
        help=(
            "Diagnostic mode: decode one view while preserving the configured "
            "artifact's view mapping"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-in-len", type=int, default=512)
    parser.add_argument("--max-queries", type=int, default=-1)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.beams <= 0:
        raise ValueError("beams must be positive")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    with open(args.idx_to_view_ids) as handle:
        idx_to_view_ids = json.load(handle)
    with open(args.eval_queries) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.max_queries > 0:
        rows = rows[: args.max_queries]
    if not rows:
        raise ValueError("evaluation query set is empty")

    n_views = validate_view_mapping(idx_to_view_ids)
    if args.single_view is not None and not 0 <= args.single_view < n_views:
        raise ValueError(f"single-view must be in [0, {n_views})")
    active_views = (
        [args.single_view] if args.single_view is not None else list(range(n_views))
    )
    if args.single_view is not None and args.calibration:
        raise ValueError("calibrated fusion is unavailable in single-view diagnostic mode")
    calibration = None
    if args.calibration:
        with open(args.calibration) as handle:
            calibration = json.load(handle)
        if calibration.get("fit_split") != "validation":
            raise ValueError("calibration artifact was not fit on validation data")
        if int(calibration.get("n_views", -1)) != n_views:
            raise ValueError("calibration view count does not match index")
    view_size = len(idx_to_view_ids[0][0].split())
    required_tokens = [f"<view_{view}>" for view in range(n_views)]
    required_tokens.extend(
        sorted({token for routes in idx_to_view_ids for route in routes for token in route.split()})
    )
    validate_atomic_tokens(tokenizer, required_tokens)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = T5ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=dtype
    ).to(args.device)
    model.eval()

    tries = build_view_tries(idx_to_view_ids, tokenizer)
    route_id_maps = build_route_id_maps(idx_to_view_ids, tokenizer)
    posting_lists = build_view_posting_lists(idx_to_view_ids)
    processors = [
        LogitsProcessorList(
            [FastRQTrieLogitsProcessor(trie, len(tokenizer), args.device)]
        )
        for trie in tries
    ]
    del tries

    cutoffs = [1, 10, 100]
    method_names = ["majority", "beam_posterior"]
    if calibration is not None:
        method_names.append(f"calibrated_{args.fusion_mode}")
    method_metric_names = ["mrr@10_doc"] + [
        f"{metric}@{cutoff}_doc"
        for cutoff in cutoffs
        for metric in ("hit", "recall", "ndcg")
    ]
    method_stats = {
        method: {metric: 0.0 for metric in method_metric_names}
        for method in method_names
    }
    view_latency = [0.0 for _ in range(n_views)]
    aggregate_counts = {
        "route_candidates": 0,
        "unique_route_candidates": 0,
        "posting_list_expanded_candidates": 0,
        "unique_document_candidates": 0,
    }
    per_view_route_candidates = [0 for _ in range(n_views)]
    candidates_handle = None
    if args.candidates_output:
        candidates_path = Path(args.candidates_output)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_handle = open(candidates_path, "w")

    warmup_count = min(max(args.warmup_queries, 0), len(rows))
    if warmup_count:
        warmup_rows = rows[:warmup_count]
        for view in active_views:
            warmup_encoded = tokenizer(
                [f"<view_{view}> {row['input']}" for row in warmup_rows],
                padding=True,
                truncation=True,
                max_length=args.max_in_len,
                return_tensors="pt",
            ).to(args.device)
            with torch.no_grad():
                model.generate(
                    **warmup_encoded,
                    num_beams=args.beams,
                    num_return_sequences=args.beams,
                    max_new_tokens=view_size + 1,
                    logits_processor=processors[view],
                    early_stopping=True,
                )
            _sync(args.device)

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        query_view_beams = [[None for _ in range(n_views)] for _ in batch]
        for view in active_views:
            inputs = [f"<view_{view}> {row['input']}" for row in batch]
            encoded = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=args.max_in_len,
                return_tensors="pt",
            ).to(args.device)
            _sync(args.device)
            tick = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    num_beams=args.beams,
                    num_return_sequences=args.beams,
                    max_new_tokens=view_size + 1,
                    logits_processor=processors[view],
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            _sync(args.device)
            view_latency[view] += time.perf_counter() - tick
            scores = generated.sequences_scores.detach().float().cpu().tolist()
            sequences = generated.sequences.detach().cpu()
            for batch_idx in range(len(batch)):
                beams = []
                for beam_idx in range(args.beams):
                    flat_idx = batch_idx * args.beams + beam_idx
                    route = generated_sequence_to_route(
                        sequences[flat_idx],
                        route_id_maps[view],
                        tokenizer.eos_token_id,
                    )
                    if route is None:
                        raise RuntimeError(
                            f"generated sequence is absent from view-{view} trie"
                        )
                    beams.append(
                        {
                            "route": route,
                            "score": float(scores[flat_idx]),
                            "rank": beam_idx + 1,
                        }
                    )
                query_view_beams[batch_idx][view] = beams
                per_view_route_candidates[view] += len(beams)

        for batch_idx, row in enumerate(batch):
            active_beams = [query_view_beams[batch_idx][view] for view in active_views]
            active_posting_lists = [posting_lists[view] for view in active_views]
            evidence, stats = aggregate_document_candidates(
                active_beams, active_posting_lists
            )
            view_log_normalizers = compute_view_route_log_normalizers(active_beams)
            ranking_limit = None if candidates_handle is not None else max(cutoffs)
            rankings = {
                "majority": rank_by_majority(evidence, top_k=ranking_limit),
                "beam_posterior": rank_by_beam_posterior(
                    evidence,
                    view_log_normalizers,
                    len(active_views),
                    top_k=ranking_limit,
                ),
            }
            if calibration is not None:
                rankings[f"calibrated_{args.fusion_mode}"] = (
                    rank_by_calibrated_fusion(
                        evidence, calibration, mode=args.fusion_mode
                    )
                )
            gold = _gold_documents(row)
            for method, ranked in rankings.items():
                query_metrics = ranking_metrics(ranked, gold, cutoffs)
                for metric, value in query_metrics.items():
                    method_stats[method][metric] += value
            for key in aggregate_counts:
                aggregate_counts[key] += stats[key]
            if candidates_handle is not None:
                majority_ranked = rankings["majority"]
                candidates_handle.write(
                    json.dumps(
                        {
                            "split_name": args.split_name,
                            "query_index": start + batch_idx,
                            "gold_document_ids": sorted(gold),
                            "ranked_document_ids_by_method": rankings,
                            "candidates": [
                                evidence[doc_idx] for doc_idx in majority_ranked
                            ],
                        }
                    )
                    + "\n"
                )

    if candidates_handle is not None:
        candidates_handle.close()
    n_queries = len(rows)
    method_metrics = {
        method: {
            metric: value / n_queries for metric, value in stats.items()
        }
        for method, stats in method_stats.items()
    }
    metrics = {
        "n_queries": n_queries,
        "split_name": args.split_name,
        "shared_model_instances": 1,
        "n_views": n_views,
        "active_views": active_views,
        "n_decoded_views": len(active_views),
        "equal_per_view_beam": args.beams,
        "total_returned_routes_per_query": args.beams * len(active_views),
        "warmup_queries_per_view": warmup_count,
        "methods": method_metrics,
        "calibration_file": args.calibration,
        "decode_latency_ms_per_query_by_view": [
            seconds / n_queries * 1000 for seconds in view_latency
        ],
        "decode_latency_ms_per_query_total": sum(view_latency) / n_queries * 1000,
        "per_view_route_candidates": per_view_route_candidates,
        "max_beam_slot_expansions": n_queries
        * len(active_views)
        * args.beams
        * (view_size + 1),
    }
    metrics.update(
        {
            f"mean_{key}_per_query": value / n_queries
            for key, value in aggregate_counts.items()
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
