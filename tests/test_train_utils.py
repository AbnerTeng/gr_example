import tempfile
from pathlib import Path

from src.train import (
    find_latest_checkpoint,
    route_view_index,
    validate_multi_view_training_contract,
)


def test_find_latest_checkpoint_uses_numeric_step_order():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for step in (900, 1000):
            checkpoint = root / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text("{}")
        assert find_latest_checkpoint(str(root)) == str(root / "checkpoint-1000")


def test_multi_view_training_requires_exactly_three_views():
    validate_multi_view_training_contract(True, 3, 9)
    for invalid_views in (1, 2, 9):
        try:
            validate_multi_view_training_contract(True, invalid_views, 9)
        except ValueError as error:
            assert "exactly 3 views" in str(error)
        else:
            raise AssertionError("non-three-view training must fail")


def test_route_view_index_enforces_global_layer_namespaces():
    assert route_view_index("<r0_1> <r1_2> <r2_3>", 3, 9) == 0
    assert route_view_index("<r3_1> <r4_2> <r5_3>", 3, 9) == 1
    assert route_view_index("<r6_1> <r7_2> <r8_3>", 3, 9) == 2
    for malformed in (
        "<r0_1> <r4_2> <r2_3>",
        "<r3_1> <r4_2>",
        "<r9_1> <r10_2> <r11_3>",
    ):
        try:
            route_view_index(malformed, 3, 9)
        except ValueError:
            pass
        else:
            raise AssertionError("cross-view or malformed route must fail")


if __name__ == "__main__":
    test_find_latest_checkpoint_uses_numeric_step_order()
    print("PASS test_find_latest_checkpoint_uses_numeric_step_order")
    test_multi_view_training_requires_exactly_three_views()
    print("PASS test_multi_view_training_requires_exactly_three_views")
    test_route_view_index_enforces_global_layer_namespaces()
    print("PASS test_route_view_index_enforces_global_layer_namespaces")
