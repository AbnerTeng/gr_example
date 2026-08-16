from transformers import AutoTokenizer

from src.eval_multi_view_gr import build_view_tries, ranking_metrics
from src.multi_view_fusion import (
    aggregate_document_candidates,
    build_view_posting_lists,
    rank_by_majority,
)
from src.multi_view_tokenizer import (
    add_multi_view_special_tokens,
    build_multi_view_special_tokens,
)


def _mapping():
    return [
        ["A", "D", "G"],
        ["A", "E", "H"],
        ["B", "D", "I"],
        ["C", "F", "G"],
    ]


def test_posting_lists_preserve_every_collision_owner():
    posting = build_view_posting_lists(_mapping())
    assert posting[0]["A"] == [0, 1]
    assert posting[1]["D"] == [0, 2]
    assert posting[2]["G"] == [0, 3]


def test_document_union_deduplicates_and_counts_distinct_views():
    posting = build_view_posting_lists(_mapping())
    beams = [
        [
            {"route": "A", "score": -1.0, "rank": 1},
            {"route": "A", "score": -1.2, "rank": 2},
            {"route": "B", "score": -2.0, "rank": 3},
        ],
        [
            {"route": "E", "score": -1.0, "rank": 1},
            {"route": "D", "score": -2.0, "rank": 2},
        ],
        [
            {"route": "H", "score": -1.0, "rank": 1},
            {"route": "I", "score": -2.0, "rank": 2},
        ],
    ]

    candidates, stats = aggregate_document_candidates(beams, posting)

    assert sorted(candidates) == [0, 1, 2]
    assert candidates[1]["hit_views"] == [0, 1, 2]
    assert candidates[1]["hit_count"] == 3
    assert candidates[1]["per_view_rank"] == {0: 1, 1: 1, 2: 1}
    assert candidates[1]["collision_size"] == {0: 2, 1: 1, 2: 1}
    assert candidates[0]["hit_count"] == 2
    assert stats == {
        "route_candidates": 7,
        "unique_route_candidates": 6,
        "posting_list_expanded_candidates": 8,
        "unique_document_candidates": 3,
    }


def test_majority_uses_complete_beams_not_only_top1_routes():
    posting = build_view_posting_lists(_mapping())
    beams = [
        [{"route": "A", "score": -1.0, "rank": 1}, {"route": "B", "score": -2.0, "rank": 2}],
        [{"route": "E", "score": -1.0, "rank": 1}, {"route": "D", "score": -2.0, "rank": 2}],
        [{"route": "H", "score": -1.0, "rank": 1}, {"route": "I", "score": -2.0, "rank": 2}],
    ]
    candidates, _ = aggregate_document_candidates(beams, posting)

    ranked = rank_by_majority(candidates)

    assert ranked[:3] == [1, 2, 0]
    assert candidates[2]["hit_count"] == 3
    assert min(candidates[2]["per_view_rank"].values()) == 2


def test_multi_gold_metrics_separate_hit_rate_from_recall_and_normalize_ndcg():
    metrics = ranking_metrics([1, 9, 2], {1, 2}, cutoffs=(1, 3))
    assert metrics["hit@1_doc"] == 1.0
    assert metrics["recall@1_doc"] == 0.5
    assert metrics["recall@3_doc"] == 1.0
    expected_ndcg = (1.0 + 1.0 / 2.0) / (1.0 + 1.0 / 1.584962500721156)
    assert abs(metrics["ndcg@3_doc"] - expected_ndcg) < 1e-12


def test_three_tries_enforce_disjoint_view_namespaces():
    tokenizer = AutoTokenizer.from_pretrained(
        "google-t5/t5-large", local_files_only=True
    )
    add_multi_view_special_tokens(
        tokenizer, build_multi_view_special_tokens(9, 16, 3)
    )
    mapping = [
        ["<r0_1> <r1_2> <r2_3>", "<r3_4> <r4_5> <r5_6>", "<r6_7> <r7_8> <r8_9>"],
        ["<r0_2> <r1_3> <r2_4>", "<r3_5> <r4_6> <r5_7>", "<r6_8> <r7_9> <r8_10>"],
    ]

    tries = build_view_tries(mapping, tokenizer)

    assert len(tries) == 3
    roots = [set(trie.allowed_flat[trie.offsets[0] : trie.offsets[1]].tolist()) for trie in tries]
    expected = [
        {tokenizer.convert_tokens_to_ids("<r0_1>"), tokenizer.convert_tokens_to_ids("<r0_2>")},
        {tokenizer.convert_tokens_to_ids("<r3_4>"), tokenizer.convert_tokens_to_ids("<r3_5>")},
        {tokenizer.convert_tokens_to_ids("<r6_7>"), tokenizer.convert_tokens_to_ids("<r6_8>")},
    ]
    assert roots == expected
    assert not (roots[0] & roots[1] or roots[1] & roots[2] or roots[0] & roots[2])


if __name__ == "__main__":
    test_posting_lists_preserve_every_collision_owner()
    print("PASS test_posting_lists_preserve_every_collision_owner")
    test_document_union_deduplicates_and_counts_distinct_views()
    print("PASS test_document_union_deduplicates_and_counts_distinct_views")
    test_majority_uses_complete_beams_not_only_top1_routes()
    print("PASS test_majority_uses_complete_beams_not_only_top1_routes")
    test_multi_gold_metrics_separate_hit_rate_from_recall_and_normalize_ndcg()
    print("PASS test_multi_gold_metrics_separate_hit_rate_from_recall_and_normalize_ndcg")
    test_three_tries_enforce_disjoint_view_namespaces()
    print("PASS test_three_tries_enforce_disjoint_view_namespaces")
