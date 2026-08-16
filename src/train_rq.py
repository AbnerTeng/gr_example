"""Train the residual quantizer that supplies nine codes for Multi-DocID."""

import json
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.rq_model import ResidualQuantizer, rq_losses, split_code_views


def build_quantizer(dim: int, cfg: DictConfig):
    if cfg.quantizer.type != "rq":
        raise ValueError(f"unsupported quantizer type: {cfg.quantizer.type}")
    return ResidualQuantizer(dim, cfg.rq.n_levels, cfg.rq.n_codes)


def make_batches(n: int, batch_size: int, rng):
    permutation = rng.permutation(n)
    return [
        permutation[start : start + batch_size]
        for start in range(0, n, batch_size)
    ]


@hydra.main(version_base=None, config_path="../configs", config_name="rq")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(f"{cfg.out_dir}/config.yaml", "w") as handle:
        handle.write(OmegaConf.to_yaml(cfg))

    embeddings = np.load(cfg.paths.doc_embeddings).astype(np.float32)
    print(f"embeddings: {embeddings.shape}")
    device = cfg.train.device
    all_embeddings = torch.from_numpy(embeddings).to(device)

    quantizer = build_quantizer(embeddings.shape[1], cfg)
    if cfg.train.kmeans_init:
        print("k-means init...")
        quantizer.init_from_kmeans(
            embeddings, cfg.train.kmeans_iter, cfg.train.seed
        )
    quantizer = quantizer.to(device)
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=cfg.train.lr)

    run = None
    if cfg.wandb.enabled:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    step = 0
    started = time.time()
    for epoch in range(cfg.train.epochs):
        quantizer.train()
        batches = make_batches(len(embeddings), cfg.train.batch_size, rng)
        epoch_logs = {}
        for indices in batches:
            batch = all_embeddings[
                torch.from_numpy(np.asarray(indices)).to(device)
            ]
            output = rq_losses(quantizer, batch, cfg.loss)
            optimizer.zero_grad()
            output["total"].backward()
            optimizer.step()

            step += 1
            logs = {
                key: value.item()
                for key, value in output.items()
                if key != "codes"
            }
            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + value
            if run and step % cfg.train.log_every == 0:
                run.log(
                    {f"train/{key}": value for key, value in logs.items()},
                    step=step,
                )

        n_batches = max(1, len(batches))
        summary = " ".join(
            f"{key}={value / n_batches:.5f}"
            for key, value in epoch_logs.items()
        )
        utilization = (quantizer.code_usage > 0).float().mean().item()
        print(
            f"epoch {epoch:3d} {summary} util={utilization:.3f} "
            f"({time.time() - started:.0f}s)"
        )
        if run:
            run.log(
                {"epoch": epoch, "train/code_util": utilization}, step=step
            )

        if (
            cfg.train.revive_dead_codes
            and (epoch + 1) % cfg.train.revive_every == 0
        ):
            sample_size = min(8192, len(embeddings))
            sample_indices = rng.choice(
                len(embeddings), sample_size, replace=False
            )
            sample = all_embeddings[
                torch.from_numpy(sample_indices).to(device)
            ]
            revived = quantizer.revive_dead_codes(sample)
            if revived:
                print(f"  revived {revived} dead codes")
        else:
            quantizer.code_usage.zero_()

    quantizer.eval()
    all_codes = []
    all_reconstructions = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 4096):
            batch = all_embeddings[start : start + 4096]
            reconstruction, codes, _, _ = quantizer.quantize(batch)
            all_codes.append(codes.cpu().numpy())
            all_reconstructions.append(reconstruction.cpu().numpy())
    codes = np.concatenate(all_codes).astype(np.int32)
    reconstructions = np.concatenate(all_reconstructions).astype(np.float32)

    np.save(f"{cfg.out_dir}/rq_codes.npy", codes)
    np.save(
        f"{cfg.out_dir}/rq_codebooks.npy",
        quantizer.codebooks.detach().cpu().numpy(),
    )
    np.save(f"{cfg.out_dir}/recon_embeddings.npy", reconstructions)
    torch.save(quantizer.state_dict(), f"{cfg.out_dir}/quantizer.pt")

    identifiers = [
        " ".join(
            f"<r{level}_{codes[index, level]}>"
            for level in range(quantizer.n_levels)
        )
        for index in range(len(codes))
    ]
    with open(f"{cfg.out_dir}/idx_to_rqid.json", "w") as handle:
        json.dump(identifiers, handle)

    if cfg.get("view_size", None) is not None:
        view_size = int(cfg.view_size)
        view_codes = split_code_views(codes, view_size)
        np.save(f"{cfg.out_dir}/view_codes.npy", view_codes)
        view_identifiers = []
        for document_views in view_codes:
            formatted = []
            for view, document_codes in enumerate(document_views):
                start_level = view * view_size
                formatted.append(
                    " ".join(
                        f"<r{start_level + offset}_{code}>"
                        for offset, code in enumerate(document_codes)
                    )
                )
            view_identifiers.append(formatted)
        with open(f"{cfg.out_dir}/idx_to_view_ids.json", "w") as handle:
            json.dump(view_identifiers, handle)

    unique = len(set(identifiers))
    print(
        f"unique docids: {unique}/{len(identifiers)} "
        f"({unique / len(identifiers) * 100:.2f}%)"
    )
    if run:
        run.log(
            {"final/unique_docid_frac": unique / len(identifiers)}, step=step
        )
        run.finish()


if __name__ == "__main__":
    main()
