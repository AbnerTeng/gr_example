"""Prepare shared-T5, view-conditioned Multi-DocID GR datasets."""

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path


RQ_TOKEN = re.compile(r"^<r(\d+)_(\d+)>$")


def _iter_jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_jsonl(path):
    return list(_iter_jsonl(path))


def iter_nq_sources(data_dir, doc_chars: int = 512):
    data_dir = Path(data_dir)

    def train_iter():
        for row in _iter_jsonl(data_dir / "train_queries.jsonl"):
            yield {
                "input": f"query: {row['query']}",
                "doc_idx": int(row["gt_doc_idx"]),
                "source": "query",
            }
        for doc_idx, row in enumerate(_iter_jsonl(data_dir / "docs.jsonl")):
            yield {
                "input": f"document: {row['text'][:doc_chars]}",
                "doc_idx": doc_idx,
                "source": "document",
            }
        pseudo_path = data_dir / "pseudo_queries.jsonl"
        if pseudo_path.exists():
            for row in _iter_jsonl(pseudo_path):
                for query in row["pseudo_queries"]:
                    if query.strip():
                        yield {
                            "input": f"query: {query.strip()}",
                            "doc_idx": int(row["doc_idx"]),
                            "source": "pseudo_query",
                        }

    def test_iter():
        for row in _iter_jsonl(data_dir / "test_queries.jsonl"):
            yield {
                "input": f"query: {row['query']}",
                "doc_idx": int(row["gt_doc_idx"]),
            }

    return train_iter(), test_iter()


def load_nq_sources(data_dir, doc_chars: int = 512):
    train, test = iter_nq_sources(data_dir, doc_chars)
    return list(train), list(test)


def iter_msmarco_sources(data_dir, doc_chars: int = 512):
    data_dir = Path(data_dir)
    corpus_path = data_dir / "corpus.jsonl"
    doc_to_idx = {}
    for doc_idx, row in enumerate(_iter_jsonl(corpus_path)):
        docid = str(row["docid"])
        if docid in doc_to_idx:
            raise ValueError(f"duplicate corpus docid {docid!r}")
        doc_to_idx[docid] = doc_idx

    def query_iter(path, include_source):
        for row in _iter_jsonl(path):
            docid = str(row["docid"])
            if docid not in doc_to_idx:
                raise ValueError(f"query target {docid!r} is absent from corpus")
            example = {
                "input": f"query: {row['query']}",
                "doc_idx": doc_to_idx[docid],
            }
            if include_source:
                example["source"] = "query"
            yield example

    def train_iter():
        yield from query_iter(data_dir / "train.jsonl", include_source=True)
        for doc_idx, row in enumerate(_iter_jsonl(corpus_path)):
            yield {
                "input": f"document: {row['document'][:doc_chars]}",
                "doc_idx": doc_idx,
                "source": "document",
            }
        pseudo_path = data_dir / "pseudo_queries.jsonl"
        if pseudo_path.exists():
            for row in _iter_jsonl(pseudo_path):
                doc_idx = int(row["doc_idx"])
                docid = str(row["docid"])
                if not 0 <= doc_idx < len(doc_to_idx):
                    raise ValueError(f"pseudo-query document index {doc_idx} is out of range")
                if doc_to_idx.get(docid) != doc_idx:
                    raise ValueError(
                        f"pseudo-query row mismatch: docid {docid!r} maps to "
                        f"{doc_to_idx.get(docid)}, not {doc_idx}"
                    )
                for query in row["pseudo_queries"]:
                    query = query.strip()
                    if query:
                        yield {
                            "input": f"query: {query}",
                            "doc_idx": doc_idx,
                            "source": "pseudo_query",
                        }

    return train_iter(), query_iter(data_dir / "dev.jsonl", include_source=False)


def load_msmarco_sources(data_dir, doc_chars: int = 512):
    train, test = iter_msmarco_sources(data_dir, doc_chars)
    return list(train), list(test)


def _is_validation_query(source, fraction: float, seed: int):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("validation fraction must be in [0, 1]")
    if source["source"] != "query" or fraction == 0.0:
        return False
    digest = hashlib.sha256(
        f"{seed}\0{source['input']}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 256) < fraction


def split_validation_queries(train_sources, fraction: float, seed: int):
    remaining = []
    validation = []
    for source in train_sources:
        if _is_validation_query(source, fraction, seed):
            validation.append(
                {"input": source["input"], "doc_idx": int(source["doc_idx"])}
            )
        else:
            remaining.append(source)
    return remaining, validation


def partition_sources_to_jsonl(train_sources, out_dir, fraction: float, seed: int):
    """Disk-backed validation partition that never materializes all source rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / ".train_sources.tmp.jsonl"
    validation_path = out_dir / ".validation_sources.tmp.jsonl"
    with open(train_path, "w") as train_handle, open(
        validation_path, "w"
    ) as validation_handle:
        for source in train_sources:
            if _is_validation_query(source, fraction, seed):
                validation_handle.write(
                    json.dumps(
                        {
                            "input": source["input"],
                            "doc_idx": int(source["doc_idx"]),
                        }
                    )
                    + "\n"
                )
            else:
                train_handle.write(json.dumps(source) + "\n")
    return train_path, validation_path


def write_multi_view_dataset(
    train_sources,
    test_sources,
    idx_to_view_ids,
    out_dir,
    validation_sources=None,
):
    validation_sources = validation_sources or []
    n_views = validate_view_mapping(idx_to_view_ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_counts = Counter()
    n_train_sources = 0
    n_train_expanded = 0

    with open(out_dir / "train.jsonl", "w") as handle:
        for source in train_sources:
            source_counts[source["source"]] += 1
            n_train_sources += 1
            for row in expand_source_example(
                source["input"], int(source["doc_idx"]), idx_to_view_ids
            ):
                handle.write(json.dumps(row) + "\n")
                n_train_expanded += 1

    n_test_sources = 0
    n_test_expanded = 0
    with open(out_dir / "test_expanded.jsonl", "w") as expanded_handle, open(
        out_dir / "eval_queries.jsonl", "w"
    ) as eval_handle:
        for source in test_sources:
            n_test_sources += 1
            doc_idx = int(source["doc_idx"])
            eval_handle.write(
                json.dumps(
                    build_eval_query(source["input"], doc_idx, idx_to_view_ids)
                )
                + "\n"
            )
            for row in expand_source_example(
                source["input"], doc_idx, idx_to_view_ids
            ):
                row["gt_rqid"] = row["output"]
                expanded_handle.write(json.dumps(row) + "\n")
                n_test_expanded += 1

    n_validation_sources = 0
    n_validation_expanded = 0
    with open(out_dir / "validation_queries.jsonl", "w") as query_handle, open(
        out_dir / "validation_expanded.jsonl", "w"
    ) as expanded_handle:
        for source in validation_sources:
            n_validation_sources += 1
            doc_idx = int(source["doc_idx"])
            query_handle.write(
                json.dumps(
                    build_eval_query(source["input"], doc_idx, idx_to_view_ids)
                )
                + "\n"
            )
            for row in expand_source_example(
                source["input"], doc_idx, idx_to_view_ids
            ):
                row["gt_rqid"] = row["output"]
                expanded_handle.write(json.dumps(row) + "\n")
                n_validation_expanded += 1

    all_view_ids = [route for views in idx_to_view_ids for route in views]
    with open(out_dir / "idx_to_view_ids.json", "w") as handle:
        json.dump(idx_to_view_ids, handle)
    with open(out_dir / "all_view_ids.json", "w") as handle:
        json.dump(all_view_ids, handle)

    manifest = {
        "n_documents": len(idx_to_view_ids),
        "n_views": n_views,
        "n_train_source_examples": n_train_sources,
        "n_train_expanded_examples": n_train_expanded,
        "n_test_source_queries": n_test_sources,
        "n_test_expanded_examples": n_test_expanded,
        "n_validation_queries": n_validation_sources,
        "n_validation_expanded_examples": n_validation_expanded,
        "train_source_counts": dict(source_counts),
        "view_markers": [f"<view_{view}>" for view in range(n_views)],
    }
    with open(out_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def validate_view_mapping(
    idx_to_view_ids, view_size: int = 3, expected_n_views: int = 3
):
    if view_size <= 0:
        raise ValueError("view_size must be positive")
    if not idx_to_view_ids:
        raise ValueError("view mapping is empty")
    n_views = len(idx_to_view_ids[0])
    if n_views != expected_n_views:
        raise ValueError(
            f"mapping has {n_views} views; expected {expected_n_views}"
        )
    for doc_idx, views in enumerate(idx_to_view_ids):
        if len(views) != n_views:
            raise ValueError(
                f"document {doc_idx} has {len(views)} views; expected {n_views}"
            )
        if len(views) == 0:
            raise ValueError(f"document {doc_idx} has no views")
        for view, route in enumerate(views):
            tokens = route.split()
            expected_levels = list(
                range(view * view_size, (view + 1) * view_size)
            )
            actual_levels = []
            for token in tokens:
                match = RQ_TOKEN.fullmatch(token)
                if match is None:
                    raise ValueError(
                        f"document {doc_idx} view {view} has invalid token {token!r}"
                    )
                actual_levels.append(int(match.group(1)))
            if actual_levels != expected_levels:
                raise ValueError(
                    f"document {doc_idx} view {view}: expected levels "
                    f"{expected_levels}, got {actual_levels}"
                )
    return n_views


def expand_source_example(input_text: str, doc_idx: int, idx_to_view_ids):
    if not 0 <= doc_idx < len(idx_to_view_ids):
        raise IndexError(f"document index {doc_idx} is outside view mapping")
    return [
        {
            "input": f"<view_{view}> {input_text}",
            "output": route,
            "doc_idx": doc_idx,
            "view": view,
        }
        for view, route in enumerate(idx_to_view_ids[doc_idx])
    ]


def build_eval_query(input_text: str, doc_idx: int, idx_to_view_ids):
    if not 0 <= doc_idx < len(idx_to_view_ids):
        raise IndexError(f"document index {doc_idx} is outside view mapping")
    return {
        "input": input_text,
        "gt_doc_idx": doc_idx,
        "gt_view_rqids": list(idx_to_view_ids[doc_idx]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nq", "msmarco300k"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--idx-to-view-ids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--doc-chars", type=int, default=512)
    parser.add_argument("--validation-query-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-sources", type=int, default=-1)
    parser.add_argument("--max-test-sources", type=int, default=-1)
    parser.add_argument("--max-validation-queries", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "nq":
        raw_train_sources, test_sources = iter_nq_sources(
            args.data_dir, args.doc_chars
        )
    else:
        raw_train_sources, test_sources = iter_msmarco_sources(
            args.data_dir, args.doc_chars
        )
    train_partition, validation_partition = partition_sources_to_jsonl(
        raw_train_sources,
        args.out_dir,
        args.validation_query_fraction,
        args.seed,
    )
    train_sources = _iter_jsonl(train_partition)
    validation_sources = _iter_jsonl(validation_partition)
    if args.max_train_sources > 0:
        train_sources = itertools.islice(train_sources, args.max_train_sources)
    if args.max_test_sources > 0:
        test_sources = itertools.islice(test_sources, args.max_test_sources)
    if args.max_validation_queries > 0:
        validation_sources = itertools.islice(
            validation_sources, args.max_validation_queries
        )
    with open(args.idx_to_view_ids) as handle:
        idx_to_view_ids = json.load(handle)
    try:
        manifest = write_multi_view_dataset(
            train_sources,
            test_sources,
            idx_to_view_ids,
            args.out_dir,
            validation_sources=validation_sources,
        )
    finally:
        train_partition.unlink(missing_ok=True)
        validation_partition.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
