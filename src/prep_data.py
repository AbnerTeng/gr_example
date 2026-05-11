import json
import random

import hydra
from omegaconf import DictConfig

from src.msmarco_utils import extract_docs_and_queries, load_jsonl


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    p = cfg.paths
    random.seed(cfg.prep_data.seed)

    print("Loading RQ docid map...")
    with open(p.semid_to_rqid) as f:
        semid_to_rqid = json.load(f)

    print("Loading MSMARCO train data...")
    raw = load_jsonl(p.msmarco_train)
    docs, train_queries, train_format = extract_docs_and_queries(raw)
    print(f"Detected MSMARCO train format: {train_format}")

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
    test_raw = load_jsonl(p.msmarco_test)
    _, test_queries, test_format = extract_docs_and_queries(test_raw)
    print(f"Detected MSMARCO test format: {test_format}")

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
            }
        )

    with open(p.test_data, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Test samples: {len(test_samples)} → {p.test_data}")


if __name__ == "__main__":
    main()
