from src.gen_pseudo_queries_msmarco300k import normalize_query as msmarco_norm
from src.gen_pseudo_queries_nq import norm as nq_norm


def test_punctuation_is_replaced_with_word_boundary():
    for normalize in (msmarco_norm, nq_norm):
        assert normalize("What-is X?") == normalize("what is x")


if __name__ == "__main__":
    test_punctuation_is_replaced_with_word_boundary()
    print("PASS test_punctuation_is_replaced_with_word_boundary")
