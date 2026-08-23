import json
import tempfile
from pathlib import Path

from src.eval_retrieval_baselines import (
    load_corpus,
    load_queries,
    ranking_metrics,
    strip_query_prefix,
)


def test_loaders_preserve_document_indices_and_gold_indices():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus = root / "corpus.jsonl"
        queries = root / "queries.jsonl"
        corpus.write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    {"docid": "external-a", "text": "alpha"},
                    {"docid": "external-b", "document": "beta"},
                ]
            )
            + "\n"
        )
        queries.write_text(
            json.dumps({"input": "query: beta question", "gt_doc_idx": 1}) + "\n"
        )
        docs, docids = load_corpus(corpus)
        loaded_queries, gold = load_queries(queries, len(docs))
        assert docs == ["alpha", "beta"]
        assert docids == ["external-a", "external-b"]
        assert loaded_queries == ["beta question"]
        assert gold == [{1}]


def test_metrics_use_document_rank_and_support_multiple_golds():
    metrics = ranking_metrics([[4, 2, 1], [7, 8, 9]], [{1, 2}, {8}])
    assert metrics["R@1"] == 0.0
    assert metrics["R@10"] == 1.0
    assert metrics["R@100"] == 1.0
    assert abs(metrics["MRR@10"] - 0.5) < 1e-12
    expected_ndcg = ((1 / 1.584962500721156 + 1 / 2) / (1 + 1 / 1.584962500721156) + 1 / 1.584962500721156) / 2
    assert abs(metrics["nDCG@10"] - expected_ndcg) < 1e-12


def test_query_prefix_is_removed_only_at_the_start():
    assert strip_query_prefix("query: What Is This?") == "What Is This?"
    assert strip_query_prefix("a query: inside") == "a query: inside"


if __name__ == "__main__":
    test_loaders_preserve_document_indices_and_gold_indices()
    print("PASS test_loaders_preserve_document_indices_and_gold_indices")
    test_metrics_use_document_rank_and_support_multiple_golds()
    print("PASS test_metrics_use_document_rank_and_support_multiple_golds")
    test_query_prefix_is_removed_only_at_the_start()
    print("PASS test_query_prefix_is_removed_only_at_the_start")
