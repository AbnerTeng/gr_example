import json

from src.prep_multi_view_gr import (
    split_full_rqids_into_views,
    validate_view_mapping,
)
from src.train import validate_multi_view_training_contract


def _full_ids():
    return [
        " ".join(f"<r{level}_{level + offset}>" for level in range(9))
        for offset in (0, 10)
    ]


def test_split_full_rqids_supports_1x9_3x3_and_9x1():
    full = _full_ids()

    one_by_nine = split_full_rqids_into_views(full, n_views=1)
    three_by_three = split_full_rqids_into_views(full, n_views=3)
    nine_by_one = split_full_rqids_into_views(full, n_views=9)

    assert one_by_nine[0] == [full[0]]
    assert three_by_three[0] == [
        "<r0_0> <r1_1> <r2_2>",
        "<r3_3> <r4_4> <r5_5>",
        "<r6_6> <r7_7> <r8_8>",
    ]
    assert nine_by_one[0] == [f"<r{level}_{level}>" for level in range(9)]

    assert validate_view_mapping(one_by_nine, n_levels=9) == 1
    assert validate_view_mapping(three_by_three, n_levels=9) == 3
    assert validate_view_mapping(nine_by_one, n_levels=9) == 9


def test_split_full_rqids_rejects_nondivisible_layout():
    try:
        split_full_rqids_into_views(_full_ids(), n_views=2)
    except ValueError as error:
        assert "divide" in str(error)
    else:
        raise AssertionError("nine RQ levels cannot be divided into two equal views")


def test_validate_view_mapping_rejects_wrong_global_layer_namespace_for_9x1():
    mapping = split_full_rqids_into_views(_full_ids(), n_views=9)
    mapping[0][4] = "<r3_4>"
    try:
        validate_view_mapping(mapping, n_levels=9)
    except ValueError as error:
        assert "document 0 view 4" in str(error)
    else:
        raise AssertionError("wrong global RQ layer must be rejected")


def test_training_contract_accepts_all_equal_contiguous_splits():
    for n_views in (1, 3, 9):
        validate_multi_view_training_contract(True, n_views, n_levels=9)


if __name__ == "__main__":
    test_split_full_rqids_supports_1x9_3x3_and_9x1()
    print("PASS test_split_full_rqids_supports_1x9_3x3_and_9x1")
    test_split_full_rqids_rejects_nondivisible_layout()
    print("PASS test_split_full_rqids_rejects_nondivisible_layout")
    test_validate_view_mapping_rejects_wrong_global_layer_namespace_for_9x1()
    print("PASS test_validate_view_mapping_rejects_wrong_global_layer_namespace_for_9x1")
    test_training_contract_accepts_all_equal_contiguous_splits()
    print("PASS test_training_contract_accepts_all_equal_contiguous_splits")
