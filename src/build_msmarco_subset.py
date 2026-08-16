"""Carve a GR-sized subset out of the full 8.8M MS MARCO passage collection.

Full 8.8M is right for tokenizer-level analysis (DocID space is 486x the corpus,
a realistic capacity ratio) but not for GR training: with pseudo-queries that is
~54M samples per epoch, roughly 10 days each.

The subset must contain every dev-qrel document, or dev recall is not measurable.
Beyond that it keeps a share of train-qrel documents (so real queries survive)
and fills the rest with unjudged passages as distractors.
"""

import argparse
import json

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/msmarco8m/corpus.jsonl")
    ap.add_argument("--train", default="data/msmarco8m/train.jsonl")
    ap.add_argument("--dev", default="data/msmarco8m/dev.jsonl")
    ap.add_argument("--out-dir", default="data/msmarco300k")
    ap.add_argument("--n-docs", type=int, default=300000)
    ap.add_argument("--train-doc-frac", type=float, default=0.5,
                    help="share of the budget spent on train-qrel documents")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    import os

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = [json.loads(l) for l in open(args.train)]
    dev = [json.loads(l) for l in open(args.dev)]
    dev_docs = {r["docid"] for r in dev}
    train_docs = {r["docid"] for r in train}
    print(f"train pairs {len(train):,} over {len(train_docs):,} docs")
    print(f"dev   pairs {len(dev):,} over {len(dev_docs):,} docs")

    # every dev gold document is mandatory
    keep = set(dev_docs)
    budget = args.n_docs - len(keep)

    # then a share of train-qrel documents, so real queries survive
    pool = sorted(train_docs - keep)
    n_train = min(len(pool), int(budget * args.train_doc_frac))
    keep |= set(rng.choice(pool, size=n_train, replace=False).tolist())
    print(f"after dev+train docs: {len(keep):,}")

    # fill with unjudged passages as distractors
    need = args.n_docs - len(keep)
    fill = []
    seen = 0
    with open(args.corpus) as f:
        for line in f:
            d = json.loads(line)
            seen += 1
            if d["docid"] not in keep:
                fill.append(d["docid"])
    rng.shuffle(fill)
    keep |= set(fill[:need])
    print(f"corpus scanned {seen:,}; final subset {len(keep):,}")

    # write corpus subset in the order it appears, with a stable index
    idx = {}
    with open(f"{args.out_dir}/corpus.jsonl", "w") as out, open(args.corpus) as f:
        for line in f:
            d = json.loads(line)
            if d["docid"] in keep:
                idx[d["docid"]] = len(idx)
                out.write(line)

    for name, rows in (("train", train), ("dev", dev)):
        kept = [r for r in rows if r["docid"] in idx]
        with open(f"{args.out_dir}/{name}.jsonl", "w") as out:
            for r in kept:
                out.write(json.dumps(r) + "\n")
        print(f"{name}: kept {len(kept):,}/{len(rows):,} pairs")

    # the statistic that decided against MS MARCO for query-side methods
    tr_docs = {r["docid"] for r in train if r["docid"] in idx}
    dv_docs = {r["docid"] for r in dev if r["docid"] in idx}
    ov = len(dv_docs & tr_docs)
    print(f"\ndev gold docs also covered by a train query: "
          f"{ov}/{len(dv_docs)} ({ov / max(len(dv_docs), 1) * 100:.1f}%)")
    print(f"DocID space 256^4 / corpus = {256 ** 4 / len(idx):,.0f}x")


if __name__ == "__main__":
    main()
