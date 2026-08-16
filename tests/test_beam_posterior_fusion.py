import math
import unittest

from src.multi_view_fusion import (
    compute_view_route_log_normalizers,
    rank_by_beam_posterior,
)


class BeamPosteriorFusionTest(unittest.TestCase):
    def test_view_local_normalization_is_invariant_to_additive_score_shift(self):
        candidates = {
            1: {
                "document_id": 1,
                "per_view_best_score": {0: -1.0, 1: 98.0},
                "per_view_rank": {0: 1, 1: 2},
                "collision_size": {0: 1, 1: 1},
            },
            2: {
                "document_id": 2,
                "per_view_best_score": {0: -2.0, 1: 99.0},
                "per_view_rank": {0: 2, 1: 1},
                "collision_size": {0: 1, 1: 1},
            },
        }
        normalizers = [
            math.log(math.exp(-1.0) + math.exp(-2.0)),
            math.log(math.exp(98.0) + math.exp(99.0)),
        ]
        shifted = {doc: {**item, "per_view_best_score": {0: item["per_view_best_score"][0], 1: item["per_view_best_score"][1] - 100.0}} for doc, item in candidates.items()}
        shifted_normalizers = [normalizers[0], normalizers[1] - 100.0]
        self.assertEqual(
            rank_by_beam_posterior(candidates, normalizers, n_views=2),
            rank_by_beam_posterior(shifted, shifted_normalizers, n_views=2),
        )

    def test_collision_mass_is_divided_among_route_owners(self):
        candidates = {
            1: {"document_id": 1, "per_view_best_score": {0: -1.0}, "per_view_rank": {0: 1}, "collision_size": {0: 10}},
            2: {"document_id": 2, "per_view_best_score": {0: -1.0}, "per_view_rank": {0: 2}, "collision_size": {0: 1}},
        }
        self.assertEqual(rank_by_beam_posterior(candidates, [0.0], n_views=1), [2, 1])

    def test_route_normalizer_deduplicates_duplicate_beam_routes(self):
        beams = [[
            {"route": "a", "score": -1.0, "rank": 1},
            {"route": "a", "score": -2.0, "rank": 2},
            {"route": "b", "score": -3.0, "rank": 3},
        ]]
        expected = math.log(math.exp(-1.0) + math.exp(-3.0))
        self.assertAlmostEqual(compute_view_route_log_normalizers(beams)[0], expected)


if __name__ == "__main__":
    unittest.main()
