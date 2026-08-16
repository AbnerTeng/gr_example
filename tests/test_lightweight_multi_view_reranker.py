import unittest
import numpy as np

from src.lightweight_multi_view_reranker import (
    evaluate_rows,
    extract_features,
    feature_names,
    fit_linear_reranker,
    rank_candidates,
    score_floors_from_rows,
    select_training_candidates,
    stable_is_dev,
)


def calibration():
    return {
        "n_views": 3,
        "views": {
            "0": {"temperature": 0.5, "bias": 0.0},
            "1": {"temperature": 1.0, "bias": 1.0},
            "2": {"temperature": 2.0, "bias": -1.0},
        },
    }


def candidate(doc_id, scores, ranks=None, collisions=None):
    ranks = ranks or {view: index + 1 for index, view in enumerate(scores)}
    collisions = collisions or {view: 1 for view in scores}
    return {
        "document_id": doc_id,
        "hit_count": len(scores),
        "per_view_best_score": scores,
        "per_view_rank": ranks,
        "collision_size": collisions,
    }


class LightweightRerankerTest(unittest.TestCase):
    def test_stable_query_split_is_deterministic_and_non_degenerate(self):
        first = [stable_is_dev(i, fraction=0.2, seed=42) for i in range(1000)]
        second = [stable_is_dev(i, fraction=0.2, seed=42) for i in range(1000)]
        self.assertEqual(first, second)
        self.assertTrue(150 <= sum(first) <= 250)

    def test_missing_view_uses_floor_and_presence_indicator_not_raw_zero(self):
        names = feature_names(3)
        values = extract_features(
            candidate(7, {"0": -2.0}, ranks={"0": 4}, collisions={"0": 3}),
            calibration(),
            score_floors=[-9.0, -8.0, -7.0],
            n_views=3,
        )
        by_name = dict(zip(names, values))
        self.assertEqual(by_name["view0_score"], -4.0)
        self.assertEqual(by_name["view0_present"], 1.0)
        self.assertEqual(by_name["view1_score"], -8.0)
        self.assertEqual(by_name["view1_present"], 0.0)
        self.assertEqual(by_name["view1_reciprocal_rank"], 0.0)
        self.assertEqual(by_name["view1_log_collision"], 0.0)

    def test_hard_negative_selection_keeps_positive_and_caps_negatives(self):
        row = {
            "gold_document_ids": [1],
            "candidates": [
                candidate(1, {"0": -0.2, "1": -0.4}),
                candidate(2, {"0": -0.1}),
                candidate(3, {"1": -0.1}),
                candidate(4, {"2": -0.1}),
            ],
        }
        selected, labels = select_training_candidates(
            row,
            calibration(),
            score_floors=[-9.0, -9.0, -9.0],
            n_views=3,
            n_negatives=2,
        )
        self.assertEqual(sum(labels), 1)
        self.assertEqual(len(labels), 3)
        self.assertIn(1, [item["document_id"] for item in selected])

    def test_linear_fit_exports_weights_that_rank_synthetic_positives_first(self):
        rows = []
        for query_index in range(20):
            rows.append(
                {
                    "query_index": query_index,
                    "gold_document_ids": [query_index * 10],
                    "candidates": [
                        candidate(query_index * 10, {"0": -0.1, "1": -0.2, "2": -0.3}),
                        candidate(query_index * 10 + 1, {"0": -3.0}),
                        candidate(query_index * 10 + 2, {"1": -4.0}),
                    ],
                }
            )
        floors = score_floors_from_rows(rows, calibration(), n_views=3)
        artifact = fit_linear_reranker(
            rows,
            calibration(),
            floors,
            n_views=3,
            n_negatives=2,
            regularization_c=1.0,
        )
        candidates = {int(c["document_id"]): c for c in rows[0]["candidates"]}
        self.assertEqual(rank_candidates(candidates, artifact)[0], 0)

    def test_evaluate_rows_reports_document_metrics(self):
        names = feature_names(3)
        coefficients = np.zeros(len(names), dtype=float)
        coefficients[names.index("view0_score")] = 1.0
        artifact = {
            "n_views": 3,
            "score_floors": [-9.0, -9.0, -9.0],
            "feature_names": names,
            "coefficients": coefficients.tolist(),
            "intercept": 0.0,
            "calibration": calibration(),
        }
        rows = [
            {
                "gold_document_ids": [10],
                "candidates": [
                    candidate(10, {"0": -0.1}),
                    candidate(20, {"0": -2.0}),
                ],
            }
        ]
        metrics = evaluate_rows(rows, artifact)
        self.assertEqual(metrics["n_queries"], 1)
        self.assertEqual(metrics["recall@1_doc"], 1.0)
        self.assertEqual(metrics["mrr@10_doc"], 1.0)

    def test_saved_linear_model_ranks_candidates_by_extracted_features(self):
        names = feature_names(3)
        coefficients = np.zeros(len(names), dtype=float)
        coefficients[names.index("view0_score")] = 1.0
        artifact = {
            "n_views": 3,
            "score_floors": [-9.0, -9.0, -9.0],
            "feature_names": names,
            "coefficients": coefficients.tolist(),
            "intercept": 0.0,
            "calibration": calibration(),
        }
        candidates = {
            10: candidate(10, {"0": -0.1}),
            20: candidate(20, {"0": -2.0}),
        }
        self.assertEqual(rank_candidates(candidates, artifact), [10, 20])


if __name__ == "__main__":
    unittest.main()
