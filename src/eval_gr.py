"""Downstream generative-retrieval evaluation.

Reports Hit/Recall@{1,10,100}, MRR@10, NDCG@{10,100}, decoding latency, the
number of distinct L1 prefixes covered by the top-K beams, and oracle prefix
(semantic-region) coverage.

Two families of retrieval metrics are reported and they mean different things:

  *_rqid : is the gold DocID string among the top-K generated DocIDs?
           Collisions are invisible here, which FLATTERS variants that collide
           more.
  *_doc  : beams are expanded into the documents that own each DocID, so a
           DocID shared by m documents contributes m candidates. This is the
           metric that actually pays for collisions, and it is the one to use
           when comparing DocID schemes with different collision rates.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.fast_trie import FastRQTrie, FastRQTrieLogitsProcessor

RQ_PATTERN = re.compile(r"<r\d+_\d+>")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--idx-to-rqid", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--beams", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-in-len", type=int, default=512)
    ap.add_argument("--max-queries", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def dcg_at(rank: int, k: int) -> float:
    """Single relevant item: DCG = 1/log2(rank+1) when rank <= k."""
    import math

    return 1.0 / math.log2(rank + 1) if 0 < rank <= k else 0.0


def main():
    args = parse_args()
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    ).to(dev)
    model.eval()

    idx_to_rqid = json.load(open(args.idx_to_rqid))
    # DocID -> owning documents (collision groups)
    rqid_to_docs = defaultdict(list)
    for i, r in enumerate(idx_to_rqid):
        rqid_to_docs[r].append(i)
    # Documents sharing a DocID are genuinely indistinguishable to the decoder,
    # so their order must not be decided by doc index (that would bias the doc
    # metrics by where the gold happens to sit). Shuffle each group once with a
    # fixed seed: an unbiased, reproducible draw of the expected rank.
    import random as _random

    _rng = _random.Random(args.seed)
    for r in rqid_to_docs:
        if len(rqid_to_docs[r]) > 1:
            _rng.shuffle(rqid_to_docs[r])

    rows = [json.loads(l) for l in open(args.test_data)]
    if args.max_queries > 0:
        rows = rows[: args.max_queries]

    trie = FastRQTrie(idx_to_rqid, tok, tok.eos_token_id)
    proc = FastRQTrieLogitsProcessor(trie, len(tok), dev)
    from transformers import LogitsProcessorList

    procs = LogitsProcessorList([proc])

    K = [1, 10, 100]
    hit_rqid = {k: 0 for k in K}
    hit_doc = {k: 0 for k in K}
    mrr10_rqid = mrr10_doc = 0.0
    ndcg = {10: 0.0, 100: 0.0}
    uniq_l1 = {10: 0.0, 100: 0.0}
    oracle_l1 = {10: 0.0, 100: 0.0}
    oracle_l2 = {10: 0.0, 100: 0.0}
    n = 0
    gen_time = 0.0

    for s in range(0, len(rows), args.batch_size):
        batch = rows[s : s + args.batch_size]
        enc = tok(
            [b["input"] for b in batch],
            padding=True,
            truncation=True,
            max_length=args.max_in_len,
            return_tensors="pt",
        ).to(dev)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=args.beams,
                num_return_sequences=args.beams,
                max_new_tokens=8,
                logits_processor=procs,
                early_stopping=True,
            )
        torch.cuda.synchronize()
        gen_time += time.time() - t0

        dec = tok.batch_decode(out, skip_special_tokens=False)
        for i, b in enumerate(batch):
            gt_rqid = b["gt_rqid"]
            gt_tokens = gt_rqid.split()
            beams = [
                " ".join(RQ_PATTERN.findall(dec[i * args.beams + j]))
                for j in range(args.beams)
            ]

            # ---- DocID-level
            rank_rqid = beams.index(gt_rqid) + 1 if gt_rqid in beams else 0
            for k in K:
                if 0 < rank_rqid <= k:
                    hit_rqid[k] += 1
            if 0 < rank_rqid <= 10:
                mrr10_rqid += 1.0 / rank_rqid

            # ---- document-level: expand each beam into its collision group
            gt_doc = None
            docs_ranked = []
            for bm in beams:
                docs_ranked.extend(rqid_to_docs.get(bm, []))
            # the gold document index: any doc owning gt_rqid that matches gt_semid
            # (test rows carry gt_semid; fall back to the first owner)
            gt_doc = b.get("gt_doc_idx")
            if gt_doc is None:
                owners = rqid_to_docs.get(gt_rqid, [])
                gt_doc = owners[0] if owners else -1
            rank_doc = docs_ranked.index(gt_doc) + 1 if gt_doc in docs_ranked else 0
            for k in K:
                if 0 < rank_doc <= k:
                    hit_doc[k] += 1
            if 0 < rank_doc <= 10:
                mrr10_doc += 1.0 / rank_doc
            for k in (10, 100):
                ndcg[k] += dcg_at(rank_doc, k)

            # ---- beam diversity / semantic-region coverage
            for k in (10, 100):
                top = beams[:k]
                l1s = {bm.split()[0] for bm in top if bm}
                uniq_l1[k] += len(l1s)
                oracle_l1[k] += 1.0 if gt_tokens[0] in l1s else 0.0
                l2s = {" ".join(bm.split()[:2]) for bm in top if len(bm.split()) >= 2}
                oracle_l2[k] += 1.0 if " ".join(gt_tokens[:2]) in l2s else 0.0
            n += 1

    res = {"n_queries": n, "beams": args.beams}
    for k in K:
        res[f"hit@{k}_rqid"] = hit_rqid[k] / n
        res[f"recall@{k}_doc"] = hit_doc[k] / n
    res["mrr@10_rqid"] = mrr10_rqid / n
    res["mrr@10_doc"] = mrr10_doc / n
    for k in (10, 100):
        res[f"ndcg@{k}_doc"] = ndcg[k] / n
        res[f"unique_l1@{k}_beams"] = uniq_l1[k] / n
        res[f"oracle_l1_coverage@{k}"] = oracle_l1[k] / n
        res[f"oracle_l2_coverage@{k}"] = oracle_l2[k] / n
    res["decode_latency_ms_per_query"] = gen_time / n * 1000
    res["decode_throughput_qps"] = n / gen_time

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    for k, v in res.items():
        print(f"{k:32s} {v:.5f}" if isinstance(v, float) else f"{k:32s} {v}")


if __name__ == "__main__":
    main()
