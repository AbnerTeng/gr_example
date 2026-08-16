import numpy as np
import torch
from omegaconf import OmegaConf

from src.rq_model import ResidualQuantizer, rq_losses, split_code_views
from src.train_rq import build_quantizer


def test_residual_quantizer_output_contract():
    torch.manual_seed(0)
    quantizer = ResidualQuantizer(dim=18, n_levels=9, n_codes=4)
    x = torch.randn(5, 18)
    reconstruction, codes, codebook_loss, commitment_loss = quantizer.quantize(x)
    assert reconstruction.shape == x.shape
    assert codes.shape == (5, 9)
    assert codebook_loss.ndim == 0
    assert commitment_loss.ndim == 0


def test_split_code_views_uses_contiguous_disjoint_groups():
    codes = np.arange(18).reshape(2, 9)
    views = split_code_views(codes, view_size=3)
    assert views.shape == (2, 3, 3)
    np.testing.assert_array_equal(views[0, 0], [0, 1, 2])
    np.testing.assert_array_equal(views[0, 1], [3, 4, 5])
    np.testing.assert_array_equal(views[0, 2], [6, 7, 8])


def test_build_quantizer_accepts_only_rq():
    cfg = OmegaConf.create(
        {"quantizer": {"type": "rq"}, "rq": {"n_levels": 9, "n_codes": 16}}
    )
    quantizer = build_quantizer(18, cfg)
    assert isinstance(quantizer, ResidualQuantizer)
    assert quantizer.n_levels == 9

    cfg.quantizer.type = "pq"
    try:
        build_quantizer(18, cfg)
    except ValueError as error:
        assert "unsupported quantizer" in str(error)
    else:
        raise AssertionError("non-RQ quantizers must be rejected")


def test_residual_quantizer_revives_unused_codes():
    torch.manual_seed(0)
    quantizer = ResidualQuantizer(dim=18, n_levels=3, n_codes=4)
    x = torch.randn(16, 18)
    revived = quantizer.revive_dead_codes(x, threshold=1)
    assert revived == 12
    assert torch.count_nonzero(quantizer.code_usage) == 0


def test_vanilla_rq_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    quantizer = ResidualQuantizer(dim=18, n_levels=3, n_codes=4)
    x = torch.randn(8, 18)
    cfg = OmegaConf.create(
        {"recon": "mse", "codebook_weight": 1.0, "commit_weight": 0.25}
    )
    losses = rq_losses(quantizer, x, cfg)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert quantizer.codebooks.grad is not None


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"PASS {name}")
