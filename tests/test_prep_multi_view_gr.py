import json
import tempfile
from pathlib import Path

from src.prep_multi_view_gr import (
    build_eval_query,
    expand_source_example,
    load_msmarco_sources,
    load_nq_sources,
    partition_sources_to_jsonl,
    split_validation_queries,
    validate_view_mapping,
    write_multi_view_dataset,
)


def _mapping():
    return [
        ["<r0_1> <r1_2> <r2_3>", "<r3_4> <r4_5> <r5_6>", "<r6_7> <r7_8> <r8_9>"],
        ["<r0_10> <r1_11> <r2_12>", "<r3_13> <r4_14> <r5_15>", "<r6_16> <r7_17> <r8_18>"],
    ]


def test_expand_source_example_creates_three_shared_model_targets():
    rows = expand_source_example("query: alpha", doc_idx=1, idx_to_view_ids=_mapping())

    assert len(rows) == 3
    assert [row["view"] for row in rows] == [0, 1, 2]
    assert [row["doc_idx"] for row in rows] == [1, 1, 1]
    assert [row["input"] for row in rows] == [
        "<view_0> query: alpha",
        "<view_1> query: alpha",
        "<view_2> query: alpha",
    ]
    assert [row["output"] for row in rows] == _mapping()[1]


def test_build_eval_query_keeps_one_document_level_record():
    row = build_eval_query("query: held out", doc_idx=0, idx_to_view_ids=_mapping())

    assert row == {
        "input": "query: held out",
        "gt_doc_idx": 0,
        "gt_view_rqids": _mapping()[0],
    }


def test_validate_view_mapping_rejects_wrong_layer_namespace():
    mapping = _mapping()
    mapping[0][1] = "<r0_4> <r1_5> <r2_6>"
    try:
        validate_view_mapping(mapping, view_size=3)
    except ValueError as error:
        assert "document 0 view 1" in str(error)
    else:
        raise AssertionError("wrong view namespace must be rejected")


def test_validate_view_mapping_requires_exactly_three_nonempty_views():
    for malformed in ([], [["<r0_1> <r1_2> <r2_3>"]]):
        try:
            validate_view_mapping(malformed, view_size=3)
        except ValueError:
            pass
        else:
            raise AssertionError("mapping must contain exactly three views")


def test_validate_view_mapping_rejects_inconsistent_view_count():
    mapping = _mapping()
    mapping[1] = mapping[1][:-1]
    try:
        validate_view_mapping(mapping, view_size=3)
    except ValueError as error:
        assert "document 1 has 2 views; expected 3" in str(error)
    else:
        raise AssertionError("inconsistent view count must be rejected")


def _write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def test_nq_adapter_uses_explicit_document_indices_for_all_sources():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_jsonl(root / "docs.jsonl", [
            {"docid": "a", "text": "alpha document"},
            {"docid": "b", "text": "beta document"},
        ])
        _write_jsonl(root / "train_queries.jsonl", [
            {"query": "alpha?", "gt_doc_idx": 0},
        ])
        _write_jsonl(root / "test_queries.jsonl", [
            {"query": "beta?", "gt_doc_idx": 1},
        ])
        _write_jsonl(root / "pseudo_queries.jsonl", [
            {"doc_idx": 1, "pseudo_queries": ["pseudo beta", ""]},
        ])

        train, test = load_nq_sources(root, doc_chars=5)

        assert train == [
            {"input": "query: alpha?", "doc_idx": 0, "source": "query"},
            {"input": "document: alpha", "doc_idx": 0, "source": "document"},
            {"input": "document: beta ", "doc_idx": 1, "source": "document"},
            {"input": "query: pseudo beta", "doc_idx": 1, "source": "pseudo_query"},
        ]
        assert test == [{"input": "query: beta?", "doc_idx": 1}]


def test_msmarco_adapter_maps_pair_docids_to_corpus_rows():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_jsonl(root / "corpus.jsonl", [
            {"docid": "b", "document": "beta"},
            {"docid": "a", "document": "alpha"},
        ])
        _write_jsonl(root / "train.jsonl", [
            {"query": "alpha?", "docid": "a"},
        ])
        _write_jsonl(root / "dev.jsonl", [
            {"query": "beta?", "docid": "b"},
        ])
        _write_jsonl(root / "pseudo_queries.jsonl", [
            {"doc_idx": 1, "docid": "a", "pseudo_queries": ["pseudo alpha", ""]},
        ])

        train, test = load_msmarco_sources(root, doc_chars=4)

        assert train == [
            {"input": "query: alpha?", "doc_idx": 1, "source": "query"},
            {"input": "document: beta", "doc_idx": 0, "source": "document"},
            {"input": "document: alph", "doc_idx": 1, "source": "document"},
            {"input": "query: pseudo alpha", "doc_idx": 1, "source": "pseudo_query"},
        ]
        assert test == [{"input": "query: beta?", "doc_idx": 0}]


def test_write_multi_view_dataset_emits_training_and_document_eval_contracts():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        train = [
            {"input": "query: a", "doc_idx": 0, "source": "query"},
            {"input": "document: b", "doc_idx": 1, "source": "document"},
        ]
        test = [{"input": "query: b", "doc_idx": 1}]

        manifest = write_multi_view_dataset(
            train,
            test,
            _mapping(),
            out,
            validation_sources=[{"input": "query: validation", "doc_idx": 0}],
        )

        train_rows = _read_jsonl(out / "train.jsonl")
        expanded_test = _read_jsonl(out / "test_expanded.jsonl")
        expanded_validation = _read_jsonl(out / "validation_expanded.jsonl")
        eval_queries = _read_jsonl(out / "eval_queries.jsonl")
        all_view_ids = json.load(open(out / "all_view_ids.json"))
        assert len(train_rows) == 6
        assert len(expanded_test) == 3
        assert len(expanded_validation) == 3
        assert len(eval_queries) == 1
        assert len(all_view_ids) == 6
        assert manifest["n_train_source_examples"] == 2
        assert manifest["n_train_expanded_examples"] == 6
        assert manifest["train_source_counts"] == {"query": 1, "document": 1}
        assert eval_queries[0]["gt_view_rqids"] == _mapping()[1]


def test_validation_split_filters_colliding_query_sources():
    train = [
        {"input": "query: held alpha", "doc_idx": 0, "source": "query"},
        {"input": "query: held  alpha", "doc_idx": 0, "source": "query"},
        {"input": "query: alpha held", "doc_idx": 0, "source": "pseudo_query"},
        {"input": "query: safe pseudo", "doc_idx": 0, "source": "pseudo_query"},
        {"input": "document: text", "doc_idx": 0, "source": "document"},
    ]
    expected_remaining = train[3:]
    expected_validation = [{"input": "query: held alpha", "doc_idx": 0}]

    remaining, validation = split_validation_queries(train, fraction=0.5, seed=42)
    assert remaining == expected_remaining
    assert validation == expected_validation

    with tempfile.TemporaryDirectory() as directory:
        train_path, validation_path = partition_sources_to_jsonl(
            iter(train), directory, fraction=0.5, seed=42
        )
        assert _read_jsonl(train_path) == expected_remaining
        assert _read_jsonl(validation_path) == expected_validation


if __name__ == "__main__":
    test_expand_source_example_creates_three_shared_model_targets()
    print("PASS test_expand_source_example_creates_three_shared_model_targets")
    test_build_eval_query_keeps_one_document_level_record()
    print("PASS test_build_eval_query_keeps_one_document_level_record")
    test_validate_view_mapping_rejects_wrong_layer_namespace()
    print("PASS test_validate_view_mapping_rejects_wrong_layer_namespace")
    test_validate_view_mapping_requires_exactly_three_nonempty_views()
    print("PASS test_validate_view_mapping_requires_exactly_three_nonempty_views")
    test_validate_view_mapping_rejects_inconsistent_view_count()
    print("PASS test_validate_view_mapping_rejects_inconsistent_view_count")
    test_nq_adapter_uses_explicit_document_indices_for_all_sources()
    print("PASS test_nq_adapter_uses_explicit_document_indices_for_all_sources")
    test_msmarco_adapter_maps_pair_docids_to_corpus_rows()
    print("PASS test_msmarco_adapter_maps_pair_docids_to_corpus_rows")
    test_write_multi_view_dataset_emits_training_and_document_eval_contracts()
    print("PASS test_write_multi_view_dataset_emits_training_and_document_eval_contracts")
    test_validation_split_filters_colliding_query_sources()
    print("PASS test_validation_split_filters_colliding_query_sources")
