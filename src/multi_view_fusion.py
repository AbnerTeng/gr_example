"""Document-level evidence aggregation for shared-T5 Multi-DocID GR."""

import math
from collections import defaultdict


def build_view_posting_lists(idx_to_view_ids):
    if not idx_to_view_ids:
        return []
    n_views = len(idx_to_view_ids[0])
    posting_lists = [defaultdict(list) for _ in range(n_views)]
    for doc_idx, routes in enumerate(idx_to_view_ids):
        if len(routes) != n_views:
            raise ValueError(
                f"document {doc_idx} has {len(routes)} views; expected {n_views}"
            )
        for view, route in enumerate(routes):
            posting_lists[view][route].append(doc_idx)
    return [dict(posting) for posting in posting_lists]


def aggregate_document_candidates(view_beams, posting_lists):
    if len(view_beams) != len(posting_lists):
        raise ValueError("view beams and posting lists must have equal length")
    evidence = {}
    raw_route_count = sum(len(beams) for beams in view_beams)
    unique_route_count = 0
    expanded_count = 0

    for view, beams in enumerate(view_beams):
        best_routes = {}
        for candidate in beams:
            route = candidate["route"]
            previous = best_routes.get(route)
            if previous is None or candidate["score"] > previous["score"]:
                best_routes[route] = candidate
        unique_route_count += len(best_routes)

        for route, candidate in best_routes.items():
            owners = posting_lists[view].get(route, [])
            expanded_count += len(owners)
            collision_size = len(owners)
            for doc_idx in owners:
                if doc_idx not in evidence:
                    evidence[doc_idx] = {
                        "document_id": doc_idx,
                        "hit_views": [],
                        "hit_count": 0,
                        "per_view_best_score": {},
                        "per_view_rank": {},
                        "per_view_route": {},
                        "collision_size": {},
                    }
                item = evidence[doc_idx]
                item["hit_views"].append(view)
                item["per_view_best_score"][view] = float(candidate["score"])
                item["per_view_rank"][view] = int(candidate["rank"])
                item["per_view_route"][view] = route
                item["collision_size"][view] = collision_size

    for item in evidence.values():
        item["hit_views"] = sorted(set(item["hit_views"]))
        item["hit_count"] = len(item["hit_views"])

    stats = {
        "route_candidates": raw_route_count,
        "unique_route_candidates": unique_route_count,
        "posting_list_expanded_candidates": expanded_count,
        "unique_document_candidates": len(evidence),
    }
    return evidence, stats


def _logsumexp(values):
    if not values:
        raise ValueError("logsumexp requires at least one value")
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def compute_view_route_log_normalizers(view_beams):
    normalizers = []
    for beams in view_beams:
        best_by_route = {}
        for candidate in beams:
            route = candidate["route"]
            previous = best_by_route.get(route)
            if previous is None or float(candidate["score"]) > previous:
                best_by_route[route] = float(candidate["score"])
        normalizers.append(_logsumexp(list(best_by_route.values())))
    return normalizers


def rank_by_beam_posterior(candidates, view_log_normalizers, n_views: int):
    if len(view_log_normalizers) != n_views:
        raise ValueError("one route normalizer is required per view")
    ranked = []
    for doc_idx, candidate in candidates.items():
        evidence = []
        for view_key, raw_score in candidate["per_view_best_score"].items():
            view = int(view_key)
            collision_size = float(candidate["collision_size"][view_key])
            evidence.append(
                float(raw_score)
                - float(view_log_normalizers[view])
                - math.log(collision_size)
            )
        score = _logsumexp(evidence) - math.log(float(n_views))
        best_rank = min(int(rank) for rank in candidate["per_view_rank"].values())
        ranked.append((doc_idx, score, best_rank))
    ranked.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [doc_idx for doc_idx, _, _ in ranked]


def rank_by_majority(candidates):
    def key(doc_idx):
        evidence = candidates[doc_idx]
        best_rank = min(evidence["per_view_rank"].values())
        return (-evidence["hit_count"], best_rank, doc_idx)

    return sorted(candidates, key=key)
