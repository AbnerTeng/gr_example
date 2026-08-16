"""Residual quantizer used to construct the nine-code Multi-DocID routes."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_code_views(codes, view_size: int):
    if codes.ndim != 2:
        raise ValueError(f"codes must have shape (documents, codes), got {codes.shape}")
    if view_size <= 0 or codes.shape[1] % view_size != 0:
        raise ValueError(
            f"{codes.shape[1]} codes cannot be split into views of size {view_size}"
        )
    return codes.reshape(codes.shape[0], codes.shape[1] // view_size, view_size)


class ResidualQuantizer(nn.Module):
    def __init__(self, dim: int, n_levels: int, n_codes: int):
        super().__init__()
        self.n_levels = n_levels
        self.n_codes = n_codes
        self.codebooks = nn.Parameter(torch.randn(n_levels, n_codes, dim) * 0.01)
        self.register_buffer("code_usage", torch.zeros(n_levels, n_codes))

    @torch.no_grad()
    def init_from_kmeans(self, x: np.ndarray, n_iter: int = 20, seed: int = 42):
        """Greedy level-wise k-means initialization."""
        import faiss

        residual = x.astype(np.float32).copy()
        for level in range(self.n_levels):
            kmeans = faiss.Kmeans(
                d=residual.shape[1],
                k=self.n_codes,
                niter=n_iter,
                seed=seed,
                verbose=False,
                nredo=1,
            )
            kmeans.train(residual)
            _, codes = kmeans.index.search(residual, 1)
            codes = codes.reshape(-1)
            centroids = kmeans.centroids
            self.codebooks.data[level] = torch.from_numpy(centroids)
            residual = residual - centroids[codes]

    def quantize(self, x: torch.Tensor):
        """Greedily quantize each residual and return reconstruction and losses."""
        residual = x
        reconstruction = torch.zeros_like(x)
        codes = []
        codebook_loss = x.new_zeros(())
        commitment_loss = x.new_zeros(())

        for level in range(self.n_levels):
            codebook = self.codebooks[level]
            distances = (
                residual.pow(2).sum(-1, keepdim=True)
                - 2 * residual @ codebook.t()
                + codebook.pow(2).sum(-1)[None, :]
            )
            indices = distances.argmin(dim=-1)
            quantized = codebook[indices]
            codebook_loss = codebook_loss + (
                quantized - residual.detach()
            ).pow(2).sum(-1).mean()
            commitment_loss = commitment_loss + (
                residual - quantized.detach()
            ).pow(2).sum(-1).mean()
            reconstruction = reconstruction + quantized
            residual = (residual - quantized).detach()
            codes.append(indices)

            if self.training:
                self.code_usage[level].index_add_(
                    0,
                    indices,
                    torch.ones_like(indices, dtype=self.code_usage.dtype),
                )

        return (
            reconstruction,
            torch.stack(codes, dim=-1),
            codebook_loss,
            commitment_loss,
        )

    @torch.no_grad()
    def revive_dead_codes(self, x: torch.Tensor, threshold: int = 1):
        """Re-seed unused codes with sampled residuals."""
        revived = 0
        residual = x
        for level in range(self.n_levels):
            codebook = self.codebooks[level]
            distances = (
                residual.pow(2).sum(-1, keepdim=True)
                - 2 * residual @ codebook.t()
                + codebook.pow(2).sum(-1)[None, :]
            )
            indices = distances.argmin(dim=-1)
            dead = torch.where(self.code_usage[level] < threshold)[0]
            if len(dead) and len(residual):
                sample = torch.randint(0, len(residual), (len(dead),), device=x.device)
                self.codebooks.data[level, dead] = residual[sample]
                revived += len(dead)
            residual = residual - codebook[indices]
        self.code_usage.zero_()
        return revived


def rq_losses(
    quantizer: ResidualQuantizer,
    x: torch.Tensor,
    cfg,
) -> Dict[str, torch.Tensor]:
    """Vanilla RQ reconstruction, codebook, and commitment objective."""
    reconstruction, codes, codebook_loss, commitment_loss = quantizer.quantize(x)
    if cfg.recon == "cosine":
        reconstruction_loss = (
            1.0 - F.cosine_similarity(reconstruction, x, dim=-1)
        ).mean()
    else:
        reconstruction_loss = (reconstruction - x).pow(2).sum(-1).mean()

    codebook_loss = codebook_loss / quantizer.n_levels
    commitment_loss = commitment_loss / quantizer.n_levels
    total = (
        reconstruction_loss
        + cfg.codebook_weight * codebook_loss
        + cfg.commit_weight * commitment_loss
    )
    return {
        "recon": reconstruction_loss,
        "codebook": codebook_loss,
        "commit": commitment_loss,
        "total": total,
        "codes": codes,
    }
