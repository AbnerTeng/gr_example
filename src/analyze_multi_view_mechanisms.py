"""CPU-only mechanism analysis for saved Multi-DocID candidate files."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .multi_view_fusion import rank_by_beam_posterior


def rank_documents_for_view(candidates, view):
    key = str(view)
    ranked = []
    for doc_id, candidate in candidates.items():
        if key not in candidate["per_view_best_score"]:
            continue
        score = float(candidate["per_view_best_score"][key]) - math.log(
            float(candidate["collision_size"][key])
        )
        route_rank = int(candidate["per_view_rank"][key])
        ranked.append((int(doc_id), score, route_rank))
    ranked.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [doc_id for doc_id, _, _ in ranked]


def reconstruct_beam_posterior_ranking(candidates, n_views=3):
    normalizers = []
    for view in range(n_views):
        key = str(view)
        route_scores = {}
        for candidate in candidates.values():
            if key not in candidate["per_view_best_score"]:
                continue
            route = candidate["per_view_route"][key]
            score = float(candidate["per_view_best_score"][key])
            route_scores[route] = max(score, route_scores.get(route, -math.inf))
        peak = max(route_scores.values())
        normalizers.append(
            peak + math.log(sum(math.exp(score - peak) for score in route_scores.values()))
        )
    return rank_by_beam_posterior(candidates, normalizers, n_views)


def classify_view_hits(candidates, gold, n_views=3):
    hit_views = []
    for view in range(n_views):
        key = str(view)
        if any(
            doc_id in gold and key in candidate["per_view_best_score"]
            for doc_id, candidate in candidates.items()
        ):
            hit_views.append(view)
    return "none" if not hit_views else "views_" + "_".join(map(str, hit_views))


def gold_rank(ranking, gold):
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in gold:
            return rank
    return None


def collision_bin(size):
    if size == 1:
        return "1"
    if size <= 4:
        return "2-4"
    if size <= 16:
        return "5-16"
    return "17+"


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_posting_sizes(mapping_path):
    mapping = json.loads(Path(mapping_path).read_text())
    n_views = len(mapping[0])
    counts = [Counter(routes[view] for routes in mapping) for view in range(n_views)]
    sizes = [
        [counts[view][routes[view]] for view in range(n_views)]
        for routes in mapping
    ]
    return sizes, n_views


def update_rank_metrics(bucket, ranks, cutoffs):
    bucket["n"] += 1
    for method, rank in ranks.items():
        if rank is not None and rank <= 10:
            bucket[f"{method}_mrr@10"] += 1.0 / rank
        for cutoff in cutoffs:
            bucket[f"{method}_recall@{cutoff}"] += float(
                rank is not None and rank <= cutoff
            )


def finalize_bucket(bucket):
    n = bucket.get("n", 0)
    return {
        key: (value / n if key != "n" and n else value)
        for key, value in bucket.items()
    }


def analyze(args):
    posting_sizes, n_views = load_posting_sizes(args.idx_to_view_ids)
    if n_views != 3:
        raise ValueError(f"expected exactly three views, found {n_views}")

    cutoffs = (1, 10, 100)
    totals = defaultdict(float)
    collision_buckets = defaultdict(lambda: defaultdict(float))
    candidate_patterns = Counter()
    topk_patterns = {cutoff: Counter() for cutoff in cutoffs}
    overlap_sums = defaultdict(float)
    query_rows = []

    with open(args.candidates) as handle:
        for line in handle:
            row = json.loads(line)
            gold = {int(doc_id) for doc_id in row["gold_document_ids"]}
            candidates = {
                int(candidate["document_id"]): candidate
                for candidate in row["candidates"]
            }
            view_rankings = [
                rank_documents_for_view(candidates, view) for view in range(n_views)
            ]
            majority = [
                int(doc_id)
                for doc_id in row["ranked_document_ids_by_method"]["majority"]
            ]
            posterior = reconstruct_beam_posterior_ranking(candidates, n_views)
            ranks = {
                **{
                    f"view{view}": gold_rank(view_rankings[view], gold)
                    for view in range(n_views)
                },
                "majority": gold_rank(majority, gold),
                "beam_posterior": gold_rank(posterior, gold),
            }
            update_rank_metrics(totals, ranks, cutoffs)

            pattern = classify_view_hits(candidates, gold, n_views)
            candidate_patterns[pattern] += 1
            view_sets = [set(ranking) for ranking in view_rankings]
            for left in range(n_views):
                others = set().union(
                    *(view_sets[right] for right in range(n_views) if right != left)
                )
                overlap_sums[f"view{left}_unique_documents"] += len(
                    view_sets[left] - others
                )
                overlap_sums[f"view{left}_candidate_documents"] += len(view_sets[left])
                for right in range(left + 1, n_views):
                    overlap_sums[f"jaccard_view{left}_view{right}"] += jaccard(
                        view_sets[left], view_sets[right]
                    )

            for cutoff in cutoffs:
                hits = [
                    ranks[f"view{view}"] is not None
                    and ranks[f"view{view}"] <= cutoff
                    for view in range(n_views)
                ]
                hit_key = "none" if not any(hits) else "views_" + "_".join(
                    str(view) for view, hit in enumerate(hits) if hit
                )
                topk_patterns[cutoff][hit_key] += 1
                any_view = any(hits)
                view0 = hits[0]
                for method in ("majority", "beam_posterior"):
                    fused_hit = (
                        ranks[method] is not None and ranks[method] <= cutoff
                    )
                    totals[f"{method}_rescue_view0_miss@{cutoff}"] += float(
                        not view0 and fused_hit
                    )
                    totals[f"{method}_failure_despite_any_view@{cutoff}"] += float(
                        any_view and not fused_hit
                    )
                totals[f"any_view_oracle_recall@{cutoff}"] += float(any_view)

            gold_collision = [
                min(posting_sizes[doc_id][view] for doc_id in gold)
                for view in range(n_views)
            ]
            bucket = collision_buckets[collision_bin(gold_collision[0])]
            update_rank_metrics(bucket, ranks, cutoffs)
            bucket["mean_view0_collision_size"] += gold_collision[0]

            query_rows.append(
                {
                    "split": row["split_name"],
                    "query_index": row["query_index"],
                    "gold_document_ids": " ".join(map(str, sorted(gold))),
                    "view0_collision_size": gold_collision[0],
                    "view1_collision_size": gold_collision[1],
                    "view2_collision_size": gold_collision[2],
                    "candidate_hit_pattern": pattern,
                    "view0_rank": ranks["view0"] or "",
                    "view1_rank": ranks["view1"] or "",
                    "view2_rank": ranks["view2"] or "",
                    "majority_rank": ranks["majority"] or "",
                    "beam_posterior_rank": ranks["beam_posterior"] or "",
                    "view0_candidates": len(view_sets[0]),
                    "view1_candidates": len(view_sets[1]),
                    "view2_candidates": len(view_sets[2]),
                    "jaccard_01": jaccard(view_sets[0], view_sets[1]),
                    "jaccard_02": jaccard(view_sets[0], view_sets[2]),
                    "jaccard_12": jaccard(view_sets[1], view_sets[2]),
                }
            )

    n = int(totals["n"])
    summary = {
        "split": query_rows[0]["split"],
        "n_queries": n,
        "candidate_file": str(args.candidates),
        "idx_to_view_ids": str(args.idx_to_view_ids),
        "ranking_semantics": {
            "per_view": "route sequence score - log(posting-list size)",
            "fusion": "equal-view beam-conditional posterior with collision split",
        },
        "metrics": finalize_bucket(totals),
        "candidate_coverage_hit_patterns": {
            key: {"count": count, "rate": count / n}
            for key, count in sorted(candidate_patterns.items())
        },
        "topk_view_hit_patterns": {
            str(cutoff): {
                key: {"count": count, "rate": count / n}
                for key, count in sorted(patterns.items())
            }
            for cutoff, patterns in topk_patterns.items()
        },
        "candidate_overlap": {
            key: value / n for key, value in sorted(overlap_sums.items())
        },
        "view0_collision_bins": {
            key: finalize_bucket(bucket)
            for key, bucket in sorted(collision_buckets.items())
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    query_output = Path(args.query_output)
    query_output.parent.mkdir(parents=True, exist_ok=True)
    with open(query_output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(query_rows)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--idx-to-view-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
