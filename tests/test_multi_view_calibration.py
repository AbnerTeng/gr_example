import math

from src.multi_view_calibration import (
    calibrated_fusion_score,
    fit_calibration_from_rows,
    rank_by_calibrated_fusion,
)


def _validation_rows():
    return [
        {
            "split_name": "validation",
            "gold_document_ids": [0],
            "candidates": [
                {"document_id": 0, "per_view_best_score": {"0": 3.0, "1": 2.0}},
                {"document_id": 1, "per_view_best_score": {"0": -2.0}},
                {"document_id": 2, "per_view_best_score": {"1": -3.0}},
            ],
        },
        {
            "split_name": "validation",
            "gold_document_ids": [3],
            "candidates": [
                {"document_id": 3, "per_view_best_score": {"0": 2.0, "1": 3.0}},
                {"document_id": 4, "per_view_best_score": {"0": -3.0}},
                {"document_id": 5, "per_view_best_score": {"1": -2.0}},
            ],
        },
    ]


def test_fit_calibration_uses_validation_candidates_only():
    calibration = fit_calibration_from_rows(_validation_rows(), n_views=2, max_iter=50)
    assert calibration["fit_split"] == "validation"
    assert set(calibration["views"]) == {"0", "1"}
    assert all(calibration["views"][str(v)]["temperature"] > 0 for v in range(2))
    assert all(math.isfinite(calibration["views"][str(v)]["bias"]) for v in range(2))

    bad = _validation_rows()
    bad[0]["split_name"] = "test"
    try:
        fit_calibration_from_rows(bad, n_views=2)
    except ValueError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("test candidates must not be accepted for calibration")


def test_missing_view_is_not_imputed_as_zero():
    calibration = {
        "views": {
            "0": {"temperature": 1.0, "bias": 0.0},
            "1": {"temperature": 1.0, "bias": 0.0},
        }
    }
    one_view = {"per_view_best_score": {0: 1.0}}
    assert calibrated_fusion_score(one_view, calibration, mode="lse") == 1.0
    assert calibrated_fusion_score(one_view, calibration, mode="sum") == 1.0


def test_calibrated_fusion_ranks_at_document_level():
    calibration = {
        "views": {
            "0": {"temperature": 1.0, "bias": 0.0},
            "1": {"temperature": 1.0, "bias": 0.0},
        }
    }
    candidates = {
        0: {"document_id": 0, "per_view_best_score": {0: 1.0}, "per_view_rank": {0: 1}},
        1: {"document_id": 1, "per_view_best_score": {0: 0.5, 1: 0.5}, "per_view_rank": {0: 2, 1: 2}},
    }
    ranked = rank_by_calibrated_fusion(candidates, calibration, mode="lse")
    assert ranked == [1, 0]


if __name__ == "__main__":
    test_fit_calibration_uses_validation_candidates_only()
    print("PASS test_fit_calibration_uses_validation_candidates_only")
    test_missing_view_is_not_imputed_as_zero()
    print("PASS test_missing_view_is_not_imputed_as_zero")
    test_calibrated_fusion_ranks_at_document_level()
    print("PASS test_calibrated_fusion_ranks_at_document_level")
