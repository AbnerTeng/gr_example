"""Evaluate a frozen lightweight Multi-DocID reranker artifact."""

import argparse
import hashlib
import json
from pathlib import Path

from .lightweight_multi_view_reranker import evaluate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-split", default="test")
    args = parser.parse_args()

    with open(args.candidates) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("split_name") != args.expected_split for row in rows):
        raise ValueError(f"all rows must use split {args.expected_split!r}")
    artifact = json.load(open(args.model))
    metrics = evaluate_rows(rows, artifact)
    digest = hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest()
    output = {
        "split": args.expected_split,
        "model": str(Path(args.model).resolve()),
        "candidates": str(Path(args.candidates).resolve()),
        "candidates_sha256": digest,
        "metrics": metrics,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
