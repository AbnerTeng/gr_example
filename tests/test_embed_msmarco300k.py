import json
import tempfile
from pathlib import Path

from src.embed_msmarco300k import (
    load_corpus,
    load_query_pairs,
    stable_docid_hash,
)


def _write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_loaders_preserve_corpus_order_and_map_query_docids():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        queries = root / "train.jsonl"
        _write_jsonl(
            corpus,
            [
                {"docid": "doc-b", "document": "second"},
                {"docid": "doc-a", "document": "first"},
            ],
        )
        _write_jsonl(
            queries,
            [
                {"query": "where is a", "docid": "doc-a"},
                {"query": "where is b", "docid": "doc-b"},
            ],
        )

        docids, texts, doc_to_idx = load_corpus(corpus)
        query_texts, query_docidx = load_query_pairs(queries, doc_to_idx)

        assert docids == ["doc-b", "doc-a"]
        assert texts == ["second", "first"]
        assert doc_to_idx == {"doc-b": 0, "doc-a": 1}
        assert query_texts == ["where is a", "where is b"]
        assert query_docidx.tolist() == [1, 0]


def test_query_loader_rejects_docids_outside_corpus():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "queries.jsonl"
        _write_jsonl(path, [{"query": "missing", "docid": "absent"}])
        try:
            load_query_pairs(path, {"present": 0})
        except ValueError as error:
            assert "absent" in str(error)
        else:
            raise AssertionError("missing query target must not be silently skipped")


def test_docid_hash_is_stable_and_order_sensitive():
    assert stable_docid_hash(["a", "b"]) == stable_docid_hash(["a", "b"])
    assert stable_docid_hash(["a", "b"]) != stable_docid_hash(["b", "a"])


if __name__ == "__main__":
    test_loaders_preserve_corpus_order_and_map_query_docids()
    print("PASS test_loaders_preserve_corpus_order_and_map_query_docids")
    test_query_loader_rejects_docids_outside_corpus()
    print("PASS test_query_loader_rejects_docids_outside_corpus")
    test_docid_hash_is_stable_and_order_sensitive()
    print("PASS test_docid_hash_is_stable_and_order_sensitive")
