import json
import tempfile
from pathlib import Path

from src.gen_pseudo_queries_msmarco300k import (
    generation_settings,
    load_resume_count,
    normalize_query,
    parse_args,
    select_queries,
)


def test_select_queries_keeps_five_unique_non_dev_queries():
    dev_exact = {normalize_query("held-out query")}
    dev_bags = {frozenset(normalize_query("held-out query").split())}
    candidates = [
        "Alpha question",
        "alpha question",
        "held-out query",
        "query held out",
        "Beta question",
        "Gamma question",
        "Delta question",
        "Epsilon question",
        "Zeta ignored",
    ]
    assert select_queries(candidates, dev_exact, dev_bags, n_queries=5) == [
        "Alpha question",
        "Beta question",
        "Gamma question",
        "Delta question",
        "Epsilon question",
    ]


def test_generation_settings_use_conservative_hard_document_fallback():
    assert generation_settings(0) == (0.95, 1.0)
    assert generation_settings(2) == (0.95, 1.0)
    assert generation_settings(3) == (0.98, 1.2)
    assert generation_settings(9) == (0.98, 1.2)


def test_default_round_budget_reaches_hard_document_fallback():
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["gen_pseudo_queries_msmarco300k"]):
        args = parse_args()
    assert args.max_rounds > 3


def test_resume_requires_contiguous_rows_matching_corpus():
    corpus = [
        {"docid": "a", "document": "A"},
        {"docid": "b", "document": "B"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "pseudo.jsonl"
        output.write_text(
            json.dumps(
                {
                    "doc_idx": 0,
                    "docid": "a",
                    "pseudo_queries": [f"q{i}" for i in range(5)],
                }
            )
            + "\n"
        )
        assert load_resume_count(output, corpus, n_queries=5) == 1

        output.write_text(
            json.dumps(
                {
                    "doc_idx": 1,
                    "docid": "b",
                    "pseudo_queries": [f"q{i}" for i in range(5)],
                }
            )
            + "\n"
        )
        try:
            load_resume_count(output, corpus, n_queries=5)
        except ValueError as error:
            assert "expected doc_idx 0" in str(error)
        else:
            raise AssertionError("non-contiguous resume output must fail")


if __name__ == "__main__":
    test_select_queries_keeps_five_unique_non_dev_queries()
    print("PASS test_select_queries_keeps_five_unique_non_dev_queries")
    test_generation_settings_use_conservative_hard_document_fallback()
    print("PASS test_generation_settings_use_conservative_hard_document_fallback")
    test_default_round_budget_reaches_hard_document_fallback()
    print("PASS test_default_round_budget_reaches_hard_document_fallback")
    test_resume_requires_contiguous_rows_matching_corpus()
    print("PASS test_resume_requires_contiguous_rows_matching_corpus")
