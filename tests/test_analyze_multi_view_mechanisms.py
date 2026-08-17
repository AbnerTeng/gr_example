from src.analyze_multi_view_mechanisms import (
    classify_view_hits,
    rank_documents_for_view,
    reconstruct_beam_posterior_ranking,
)


def synthetic_candidates():
    return {
        0: {
            "document_id": 0,
            "hit_views": [0, 1],
            "hit_count": 2,
            "per_view_best_score": {"0": -0.1, "1": -0.1},
            "per_view_rank": {"0": 1, "1": 1},
            "per_view_route": {"0": "v0_shared", "1": "v1_gold"},
            "collision_size": {"0": 2, "1": 1},
        },
        1: {
            "document_id": 1,
            "hit_views": [0],
            "hit_count": 1,
            "per_view_best_score": {"0": -0.1},
            "per_view_rank": {"0": 1},
            "per_view_route": {"0": "v0_shared"},
            "collision_size": {"0": 2},
        },
        2: {
            "document_id": 2,
            "hit_views": [1],
            "hit_count": 1,
            "per_view_best_score": {"1": -1.0},
            "per_view_rank": {"1": 2},
            "per_view_route": {"1": "v1_other"},
            "collision_size": {"1": 1},
        },
    }


def test_collision_adjusted_view_ranking_and_rescue_pattern():
    candidates = synthetic_candidates()
    assert rank_documents_for_view(candidates, 0) == [0, 1]
    assert rank_documents_for_view(candidates, 1) == [0, 2]
    assert classify_view_hits(candidates, {0}) == "views_0_1"


def test_beam_posterior_rewards_cross_view_support():
    ranked = reconstruct_beam_posterior_ranking(synthetic_candidates(), n_views=2)
    assert ranked[0] == 0


if __name__ == "__main__":
    test_collision_adjusted_view_ranking_and_rescue_pattern()
    print("PASS test_collision_adjusted_view_ranking_and_rescue_pattern")
    test_beam_posterior_rewards_cross_view_support()
    print("PASS test_beam_posterior_rewards_cross_view_support")
