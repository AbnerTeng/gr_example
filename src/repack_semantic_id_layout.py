"""Repack a prepared Multi-DocID dataset into another contiguous RQ layout."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from .prep_multi_view_gr import (
    _iter_jsonl,
    split_full_rqids_into_views,
    write_multi_view_dataset,
)


def _strip_expected_view_marker(input_text: str, view: int) -> str:
    prefix = f"<view_{view}> "
    if not input_text.startswith(prefix):
        raise ValueError(f"input is missing expected {prefix.strip()} marker")
    return input_text[len(prefix) :]


def iter_prepared_train_sources(path, source_n_views: int):
    """Collapse contiguous view-expanded rows back to source-level records."""
    if source_n_views <= 0:
        raise ValueError("source_n_views must be positive")
    rows = _iter_jsonl(path)
    for group_idx, first in enumerate(rows):
        group = [first, *itertools.islice(rows, source_n_views - 1)]
        if len(group) != source_n_views:
            raise ValueError(
                f"source group {group_idx} is incomplete: "
                f"got {len(group)} of {source_n_views} rows"
            )
        try:
            inputs = [
                _strip_expected_view_marker(row["input"], view)
                for view, row in enumerate(group)
            ]
            views = [int(row["view"]) for row in group]
            doc_indices = [int(row["doc_idx"]) for row in group]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"source group {group_idx} is malformed: {error}") from error
        if views != list(range(source_n_views)):
            raise ValueError(
                f"source group {group_idx} has view order {views}; "
                f"expected {list(range(source_n_views))}"
            )
        if len(set(inputs)) != 1 or len(set(doc_indices)) != 1:
            raise ValueError(
                f"source group {group_idx} does not share one input and doc_idx"
            )
        yield {
            "input": inputs[0],
            "doc_idx": doc_indices[0],
            "source": "inherited_prepared_source",
        }


def iter_prepared_eval_sources(path):
    for row_idx, row in enumerate(_iter_jsonl(path)):
        try:
            yield {"input": row["input"], "doc_idx": int(row["gt_doc_idx"])}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"evaluation row {row_idx} is malformed: {error}") from error


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--full-rqids", type=Path, required=True)
    parser.add_argument("--n-views", type=int, required=True)
    parser.add_argument("--source-n-views", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output {args.out_dir}")
    with open(args.full_rqids) as handle:
        full_rqids = json.load(handle)
    idx_to_view_ids = split_full_rqids_into_views(full_rqids, args.n_views)
    n_levels = len(full_rqids[0].split())
    source_manifest_path = args.source_dir / "manifest.json"
    with open(source_manifest_path) as handle:
        source_manifest = json.load(handle)

    manifest = write_multi_view_dataset(
        iter_prepared_train_sources(
            args.source_dir / "train.jsonl", args.source_n_views
        ),
        iter_prepared_eval_sources(args.source_dir / "eval_queries.jsonl"),
        idx_to_view_ids,
        args.out_dir,
        validation_sources=iter_prepared_eval_sources(
            args.source_dir / "validation_queries.jsonl"
        ),
    )
    manifest.update(
        {
            "layout_notation": f"{args.n_views}x{n_levels // args.n_views}",
            "n_rq_levels": n_levels,
            "source_prepared_dir": str(args.source_dir),
            "source_n_views": args.source_n_views,
            "source_manifest_sha256": _sha256(source_manifest_path),
            "full_rqids_sha256": _sha256(args.full_rqids),
            "inherited_train_source_counts": source_manifest.get(
                "train_source_counts", {}
            ),
        }
    )
    with open(args.out_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
