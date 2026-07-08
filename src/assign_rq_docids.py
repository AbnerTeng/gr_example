import json

import faiss
import hydra
import numpy as np
from omegaconf import DictConfig

from src.msmarco_utils import load_docs_and_queries_by_split


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    p, rq, ard = cfg.paths, cfg.rq, cfg.assign_rq_docids

    print("Loading doc embeddings...")
    doc_embs = np.load(p.doc_embeddings).astype(np.float32)  # (N, D), L2-normalized
    print(f"  shape: {doc_embs.shape}")

    split_files = [
        ("train", p.msmarco_train),
        ("valid", p.msmarco_valid),
        ("test", p.msmarco_test),
    ]
    print("Loading doc semids from MSMARCO (train + valid + test docs)...")
    docs, queries_by_split, format_by_split = load_docs_and_queries_by_split(split_files)
    for split_name, _ in split_files:
        print(
            f"  {split_name}: format={format_by_split[split_name]}, "
            f"queries={len(queries_by_split[split_name])}"
        )
    doc_semids = [d["doc_id"] for d in docs]
    assert len(doc_semids) == doc_embs.shape[0], "Count mismatch!"

    print(f"Running {rq.n_levels}-level RQ (n_codes={rq.n_codes}) with FAISS KMeans...")
    codebooks = []
    rq_codes = np.zeros((len(doc_embs), rq.n_levels), dtype=np.int32)
    residuals = doc_embs.copy()

    for level in range(rq.n_levels):
        print(f"  Level {level}: FAISS KMeans...")
        km = faiss.Kmeans(
            d=residuals.shape[1],
            k=rq.n_codes,
            niter=ard.n_iter,
            seed=ard.seed,
            verbose=False,
            nredo=1,
        )
        km.train(residuals)
        _, codes = km.index.search(residuals, 1)
        codes = codes.reshape(-1)
        rq_codes[:, level] = codes
        centroids = km.centroids
        residuals = residuals - centroids[codes]
        codebooks.append(centroids)
        print(f"    residual norm={np.linalg.norm(residuals, axis=1).mean():.4f}")

    np.save(f"{p.data_dir}/rq_codes.npy", rq_codes)
    np.save(
        f"{p.data_dir}/rq_codebooks.npy", np.stack(codebooks)
    )  # (n_levels, n_codes, D)
    print("Saved rq_codes.npy and rq_codebooks.npy")

    semid_to_rqid = {}
    idx_to_rqid = []
    for i, semid in enumerate(doc_semids):
        rqid = " ".join(
            f"<r{level}_{rq_codes[i, level]}>" for level in range(rq.n_levels)
        )
        semid_to_rqid[semid] = rqid
        idx_to_rqid.append(rqid)

    with open(p.semid_to_rqid, "w") as f:
        json.dump(semid_to_rqid, f)
    with open(p.idx_to_rqid, "w") as f:
        json.dump(idx_to_rqid, f)

    unique = len(set(idx_to_rqid))
    print(
        f"Unique RQ docids: {unique} / {len(idx_to_rqid)} ({unique / len(idx_to_rqid) * 100:.1f}%)"
    )
    print("Done.")


if __name__ == "__main__":
    main()
