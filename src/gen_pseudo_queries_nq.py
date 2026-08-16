"""Generate pseudo-queries for NQ320k with docTTTTTquery, then drop any that
collide with a held-out test query.

Two reasons this matters here:
  * The GR pipeline needs query-like document supervision.
  * Content-derived pseudo-queries cover every document rather than only the
    documents that have a real training query.

The filtering is not optional: docTTTTTquery is trained to imitate real queries
and reproduces a fair share of the test set verbatim, which would put the exact
test question and its correct document into training.
"""

import argparse
import json
import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/nq320k/docs.jsonl")
    ap.add_argument("--test-queries", default="data/nq320k/test_queries.jsonl")
    ap.add_argument("--output", default="data/nq320k/pseudo_queries.jsonl")
    ap.add_argument("--model", default="doc2query/msmarco-t5-base-v1")
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-doc-len", type=int, default=256)
    ap.add_argument("--max-query-len", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def set_generation_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.strip().lower()).split())


@torch.no_grad()
def main():
    args = parse_args()
    set_generation_seed(args.seed)
    docs = [json.loads(l) for l in open(args.docs)]
    test = [json.loads(l) for l in open(args.test_queries)]
    test_exact = {norm(q["query"]) for q in test}
    test_bag = {frozenset(norm(q["query"]).split()) for q in test}
    print(f"docs {len(docs):,}  test queries {len(test):,}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device).eval()

    kept = dropped = 0
    with open(args.output, "w") as out:
        for i in tqdm(range(0, len(docs), args.batch_size), desc="pseudo-queries"):
            batch = docs[i : i + args.batch_size]
            enc = tok([d["text"] for d in batch], padding=True, truncation=True,
                      max_length=args.max_doc_len, return_tensors="pt").to(args.device)
            gen = model.generate(**enc, max_length=args.max_query_len, do_sample=True,
                                 top_p=0.95, num_return_sequences=args.n_queries)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            for j, d in enumerate(batch):
                qs = dec[j * args.n_queries : (j + 1) * args.n_queries]
                keep = []
                for q in qs:
                    n = norm(q)
                    if not n:
                        continue
                    if n in test_exact or frozenset(n.split()) in test_bag:
                        dropped += 1
                    else:
                        keep.append(q)
                        kept += 1
                out.write(json.dumps({"doc_idx": i + j, "docid": d["docid"],
                                      "pseudo_queries": keep}) + "\n")

    total = kept + dropped
    print(f"kept {kept:,} / dropped {dropped:,} ({dropped / max(total,1) * 100:.2f}% "
          f"collided with a test query)")


if __name__ == "__main__":
    main()
