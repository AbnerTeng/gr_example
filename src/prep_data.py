import json
import random

import hydra
from omegaconf import DictConfig

from src.msmarco_utils import load_docs_and_queries_by_split


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    p = cfg.paths
    random.seed(cfg.prep_data.seed)

    print("Loading RQ docid map...")
    with open(p.semid_to_rqid) as f:
        semid_to_rqid = json.load(f)

    split_files = [
        ("train", p.msmarco_train),
        ("valid", p.msmarco_valid),
        ("test", p.msmarco_test),
    ]
    print("Loading MSMARCO splits...")
    docs, queries_by_split, format_by_split = load_docs_and_queries_by_split(split_files)
    for split_name, _ in split_files:
        print(
            f"  {split_name}: format={format_by_split[split_name]}, "
            f"queries={len(queries_by_split[split_name])}"
        )

    train_queries = queries_by_split["train"] + queries_by_split["valid"]
    test_queries = queries_by_split["test"]
    print(
        f"Using train queries from train+valid: {len(train_queries)}, "
        f"test queries from test: {len(test_queries)}"
    )

    samples, skipped = [], 0

    # query -> rqdocid
    for d in train_queries:
        semid = d["doc_id"]
        if semid not in semid_to_rqid:
            skipped += 1
            continue
        samples.append({"input": f"query: {d['text']}", "output": semid_to_rqid[semid]})
    print(f"  Queries: {len(samples)} (skipped {skipped})")

    # document text -> rqdocid
    n_docs = 0
    for d in docs:
        semid = d["doc_id"]
        if semid not in semid_to_rqid:
            continue
        samples.append(
            {"input": f"document: {d['text']}", "output": semid_to_rqid[semid]}
        )
        n_docs += 1
    print(f"  Docs: {n_docs}")

    # pseudo-queries -> rqdocid
    n_pq = 0
    with open(p.pseudo_queries) as f:
        for line in f:
            entry = json.loads(line)
            semid = entry["doc_id"]
            if semid not in semid_to_rqid:
                continue
            rqid = semid_to_rqid[semid]
            for pq in entry["pseudo_queries"]:
                if pq.strip():
                    samples.append({"input": f"query: {pq.strip()}", "output": rqid})
                    n_pq += 1
    print(f"  Pseudo-queries: {n_pq}")

    random.shuffle(samples)
    print(f"Total training samples: {len(samples)}")

    with open(p.train_data, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved train → {p.train_data}")

    print("Building test set...")

    # doc_id -> corpus index, so downstream eval can score at document level
    semid_to_idx = {d["doc_id"]: i for i, d in enumerate(docs)}

    test_samples = []
    for d in test_queries:
        semid = d["doc_id"]
        if semid not in semid_to_rqid:
            continue
        test_samples.append(
            {
                "input": f"query: {d['text']}",
                "output": semid_to_rqid[semid],
                "gt_semid": semid,
                "gt_rqid": semid_to_rqid[semid],
                "gt_doc_idx": semid_to_idx[semid],
            }
        )

    with open(p.test_data, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Test samples: {len(test_samples)} → {p.test_data}")


if __name__ == "__main__":
    main()
