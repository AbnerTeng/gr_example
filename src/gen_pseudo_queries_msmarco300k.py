"""Generate exactly five filtered docT5query pseudo-queries per MSMARCO-300K document."""

import argparse
import itertools
import json
import os
import random
import re
from pathlib import Path


_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def normalize_query(text: str) -> str:
    return " ".join(_NON_ALNUM.sub(" ", text.strip().lower()).split())


def select_queries(candidates, dev_exact, dev_bags, n_queries: int):
    selected = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_query(candidate)
        if not normalized or normalized in seen:
            continue
        bag = frozenset(normalized.split())
        if normalized in dev_exact or bag in dev_bags:
            continue
        seen.add(normalized)
        selected.append(candidate.strip())
        if len(selected) == n_queries:
            break
    return selected


def _iter_jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_resume_count(
    output_path,
    corpus,
    n_queries: int,
    dev_exact=None,
    dev_bags=None,
) -> int:
    """Validate a partial output as a filtered contiguous corpus prefix."""
    output_path = Path(output_path)
    dev_exact = set() if dev_exact is None else dev_exact
    dev_bags = set() if dev_bags is None else dev_bags
    if not output_path.exists():
        return 0
    corpus_iter = _iter_jsonl(corpus) if isinstance(corpus, (str, Path)) else iter(corpus)
    count = 0
    with open(output_path) as handle:
        for expected_idx, line in enumerate(handle):
            try:
                corpus_row = next(corpus_iter)
            except StopIteration as error:
                raise ValueError("resume output contains more rows than the corpus") from error
            row = json.loads(line)
            if int(row["doc_idx"]) != expected_idx:
                raise ValueError(
                    f"resume row {expected_idx}: expected doc_idx {expected_idx}, "
                    f"got {row['doc_idx']}"
                )
            if str(row["docid"]) != str(corpus_row["docid"]):
                raise ValueError(
                    f"resume row {expected_idx}: docid does not match corpus"
                )
            queries = row["pseudo_queries"]
            if len(queries) != n_queries:
                raise ValueError(
                    f"resume row {expected_idx}: expected {n_queries} pseudo-queries"
                )
            normalized_queries = [normalize_query(query) for query in queries]
            if any(not query for query in normalized_queries):
                raise ValueError(
                    f"resume row {expected_idx}: pseudo-query normalizes to empty"
                )
            if len(set(normalized_queries)) != n_queries:
                raise ValueError(f"resume row {expected_idx}: pseudo-queries are not unique")
            if any(
                query in dev_exact or frozenset(query.split()) in dev_bags
                for query in normalized_queries
            ):
                raise ValueError(
                    f"resume row {expected_idx}: pseudo-query collides with dev query"
                )
            count += 1
    return count


def load_dev_collisions(dev_path):
    exact = set()
    bags = set()
    for row in _iter_jsonl(dev_path):
        normalized = normalize_query(row["query"])
        if normalized:
            exact.add(normalized)
            bags.add(frozenset(normalized.split()))
    return exact, bags


def batched(iterable, size):
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


def generation_settings(round_idx: int):
    if round_idx < 3:
        return 0.95, 1.0
    return 0.98, 1.2


def generate_candidates(model, tokenizer, texts, args, top_p=None, temperature=None):
    import torch

    top_p = args.top_p if top_p is None else top_p
    temperature = args.temperature if temperature is None else temperature

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_doc_len,
        return_tensors="pt",
    ).to(args.device)
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_length=args.max_query_len,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=args.candidates_per_round,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [
        decoded[i * args.candidates_per_round : (i + 1) * args.candidates_per_round]
        for i in range(len(texts))
    ]


def generate_exact_queries(model, tokenizer, rows, dev_exact, dev_bags, args):
    accumulated = [[] for _ in rows]
    selected = [[] for _ in rows]
    unresolved = list(range(len(rows)))
    for round_idx in range(args.max_rounds):
        if not unresolved:
            break
        random.seed(args.seed + int(rows[0][0]) * 1009 + round_idx)
        try:
            import numpy as np
            import torch

            np.random.seed((args.seed + int(rows[0][0]) * 1009 + round_idx) % (2**32))
            torch.manual_seed(args.seed + int(rows[0][0]) * 1009 + round_idx)
        except ImportError:
            pass
        top_p, temperature = generation_settings(round_idx)
        candidates = generate_candidates(
            model,
            tokenizer,
            [rows[index][1]["document"] for index in unresolved],
            args,
            top_p=top_p,
            temperature=temperature,
        )
        next_unresolved = []
        for local_index, row_index in enumerate(unresolved):
            accumulated[row_index].extend(candidates[local_index])
            selected[row_index] = select_queries(
                accumulated[row_index], dev_exact, dev_bags, args.n_queries
            )
            if len(selected[row_index]) != args.n_queries:
                next_unresolved.append(row_index)
        unresolved = next_unresolved
    if unresolved:
        failed = [int(rows[index][0]) for index in unresolved]
        raise RuntimeError(
            f"could not retain {args.n_queries} unique non-dev pseudo-queries for "
            f"documents {failed[:10]} after {args.max_rounds} rounds"
        )
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/msmarco300k/corpus.jsonl"))
    parser.add_argument("--dev-queries", type=Path, default=Path("data/msmarco300k/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/msmarco300k/pseudo_queries.jsonl"))
    parser.add_argument("--model", default="doc2query/msmarco-t5-base-v1")
    parser.add_argument("--n-queries", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-doc-len", type=int, default=256)
    parser.add_argument("--max-query-len", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=int, default=-1)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_queries <= 0 or args.candidates_per_round < args.n_queries:
        raise ValueError("candidates-per-round must be at least n-queries > 0")
    if args.batch_size <= 0 or args.max_rounds <= 0:
        raise ValueError("batch-size and max-rounds must be positive")

    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    dev_exact, dev_bags = load_dev_collisions(args.dev_queries)
    resume_count = 0 if args.no_resume else load_resume_count(
        args.output,
        args.corpus,
        args.n_queries,
        dev_exact=dev_exact,
        dev_bags=dev_bags,
    )
    if args.no_resume and args.output.exists():
        args.output.unlink()

    metadata = {
        "corpus": str(args.corpus),
        "dev_queries": str(args.dev_queries),
        "output": str(args.output),
        "generator": args.model,
        "generation_config": {
            "n_queries": args.n_queries,
            "candidates_per_round": args.candidates_per_round,
            "max_rounds": args.max_rounds,
            "max_doc_len": args.max_doc_len,
            "max_query_len": args.max_query_len,
            "do_sample": True,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "hard_document_fallback": {
                "after_round": 3,
                "top_p": 0.98,
                "temperature": 1.2,
            },
            "seed": args.seed,
        },
        "filtering": ["empty", "within_document_normalized_duplicate", "dev_exact", "dev_bag_of_words"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device).eval()

    indexed_corpus = enumerate(_iter_jsonl(args.corpus))
    remaining = itertools.islice(indexed_corpus, resume_count, None)
    if args.max_documents > 0:
        remaining = itertools.islice(remaining, args.max_documents)

    mode = "a" if resume_count else "w"
    processed = 0
    with open(args.output, mode) as output_handle:
        progress = tqdm(desc="docT5query", initial=resume_count, unit="doc")
        for rows in batched(remaining, args.batch_size):
            selections = generate_exact_queries(
                model, tokenizer, rows, dev_exact, dev_bags, args
            )
            for (doc_idx, row), queries in zip(rows, selections):
                output_handle.write(
                    json.dumps(
                        {
                            "doc_idx": doc_idx,
                            "docid": str(row["docid"]),
                            "pseudo_queries": queries,
                            "generator": args.model,
                        }
                    )
                    + "\n"
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
            processed += len(rows)
            progress.update(len(rows))
        progress.close()
    final_count = resume_count + processed
    print(f"Saved {processed:,} new rows; contiguous output now has {final_count:,} rows → {args.output}")


if __name__ == "__main__":
    main()
