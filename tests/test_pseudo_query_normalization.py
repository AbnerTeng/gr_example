import torch

from src.gen_pseudo_queries_msmarco300k import normalize_query as msmarco_norm
from src.gen_pseudo_queries_nq import norm as nq_norm, set_generation_seed


def test_punctuation_is_replaced_with_word_boundary():
    for normalize in (msmarco_norm, nq_norm):
        assert normalize("What-is X?") == normalize("what is x")


def test_torch_sampling_seed_is_reproducible():
    set_generation_seed(42)
    first = torch.rand(4)
    set_generation_seed(42)
    second = torch.rand(4)
    assert torch.equal(first, second)


if __name__ == "__main__":
    test_punctuation_is_replaced_with_word_boundary()
    print("PASS test_punctuation_is_replaced_with_word_boundary")
    test_torch_sampling_seed_is_reproducible()
    print("PASS test_torch_sampling_seed_is_reproducible")
