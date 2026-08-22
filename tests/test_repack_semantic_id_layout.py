import json
import tempfile
from pathlib import Path

from src.repack_semantic_id_layout import iter_prepared_train_sources


def _write_rows(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_iter_prepared_train_sources_recovers_one_source_from_three_views():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        _write_rows(
            path,
            [
                {"input": "<view_0> query: alpha", "output": "<r0_1> <r1_2> <r2_3>", "doc_idx": 7, "view": 0},
                {"input": "<view_1> query: alpha", "output": "<r3_4> <r4_5> <r5_6>", "doc_idx": 7, "view": 1},
                {"input": "<view_2> query: alpha", "output": "<r6_7> <r7_8> <r8_9>", "doc_idx": 7, "view": 2},
            ],
        )

        assert list(iter_prepared_train_sources(path, source_n_views=3)) == [
            {"input": "query: alpha", "doc_idx": 7, "source": "inherited_prepared_source"}
        ]


def test_iter_prepared_train_sources_rejects_cross_view_source_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        _write_rows(
            path,
            [
                {"input": "<view_0> query: alpha", "output": "<r0_1>", "doc_idx": 7, "view": 0},
                {"input": "<view_1> query: beta", "output": "<r1_2>", "doc_idx": 7, "view": 1},
                {"input": "<view_2> query: alpha", "output": "<r2_3>", "doc_idx": 7, "view": 2},
            ],
        )

        try:
            list(iter_prepared_train_sources(path, source_n_views=3))
        except ValueError as error:
            assert "source group 0" in str(error)
        else:
            raise AssertionError("cross-view source mismatch must be rejected")


if __name__ == "__main__":
    test_iter_prepared_train_sources_recovers_one_source_from_three_views()
    print("PASS test_iter_prepared_train_sources_recovers_one_source_from_three_views")
    test_iter_prepared_train_sources_rejects_cross_view_source_mismatch()
    print("PASS test_iter_prepared_train_sources_rejects_cross_view_source_mismatch")
