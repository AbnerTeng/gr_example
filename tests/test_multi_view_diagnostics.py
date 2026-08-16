import numpy as np

from src.eval_multi_view_codes import compute_code_diagnostics


def test_compute_code_diagnostics_tracks_complementary_collisions():
    codes = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ],
        dtype=np.int64,
    )

    metrics = compute_code_diagnostics(codes, view_size=3)

    assert metrics["views"][0]["unique_code_ratio"] == 0.5
    assert metrics["views"][0]["collided_document_fraction"] == 1.0
    assert metrics["views"][0]["max_collision_group"] == 2
    assert metrics["views"][1]["unique_code_ratio"] == 0.75
    assert metrics["views"][1]["collided_document_fraction"] == 0.5
    assert metrics["oracle_any_singleton_document_fraction"] == 0.75
    assert metrics["all_views_collided_document_fraction"] == 0.25


if __name__ == "__main__":
    test_compute_code_diagnostics_tracks_complementary_collisions()
    print("PASS test_compute_code_diagnostics_tracks_complementary_collisions")
