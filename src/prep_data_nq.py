"""Build GR training data for NQ320k in the format src/train.py expects.

Pseudo-queries were already filtered against the test set at generation time
(src/gen_pseudo_queries_nq.py), so nothing here can leak.
"""

import argparse
import json
import os
import random


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq-dir", default="data/nq320k")
    ap.add_argument("--idx-to-rqid", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--doc-chars", type=int, default=512,
                    help="chars of document text for the indexing task. NQ pages are "
                         "~500 tokens; with mixed batches every batch pads to 512, "
                         "wasting ~7x compute since 88 percent of samples are 13-token "
                         "queries. DSI-style direct indexing on leading tokens is "
                         "standard, and an NQ page opens with its title and lead section.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    idx_to_rqid = json.load(open(args.idx_to_rqid))
    docs = [json.loads(l) for l in open(f"{args.nq_dir}/docs.jsonl")]
    assert len(docs) == len(idx_to_rqid), f"{len(docs)} vs {len(idx_to_rqid)}"
    semid_to_rqid = {d["docid"]: idx_to_rqid[i] for i, d in enumerate(docs)}

    samples = []
    tq = [json.loads(l) for l in open(f"{args.nq_dir}/train_queries.jsonl")]
    for r in tq:
        samples.append({"input": f"query: {r['query']}",
                        "output": idx_to_rqid[r["gt_doc_idx"]]})
    for i, d in enumerate(docs):
        samples.append({"input": f"document: {d['text'][: args.doc_chars]}",
                        "output": idx_to_rqid[i]})
    n_pq = 0
    for line in open(f"{args.nq_dir}/pseudo_queries.jsonl"):
        e = json.loads(line)
        for q in e["pseudo_queries"]:
            if q.strip():
                samples.append({"input": f"query: {q.strip()}",
                                "output": idx_to_rqid[e["doc_idx"]]})
                n_pq += 1
    random.shuffle(samples)
    print(f"  queries={len(tq):,} docs={len(docs):,} pseudo={n_pq:,} total={len(samples):,}")
    with open(f"{args.out_dir}/train.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    test = []
    for r in (json.loads(l) for l in open(f"{args.nq_dir}/test_queries.jsonl")):
        rq = idx_to_rqid[r["gt_doc_idx"]]
        test.append({"input": f"query: {r['query']}", "output": rq,
                     "gt_semid": r["docid"], "gt_rqid": rq,
                     "gt_doc_idx": r["gt_doc_idx"]})
    with open(f"{args.out_dir}/test.jsonl", "w") as f:
        for s in test:
            f.write(json.dumps(s) + "\n")
    json.dump(idx_to_rqid, open(f"{args.out_dir}/idx_to_rqid.json", "w"))
    json.dump(semid_to_rqid, open(f"{args.out_dir}/semid_to_rqid.json", "w"))
    print(f"  test={len(test):,} -> {args.out_dir}")


if __name__ == "__main__":
    main()
