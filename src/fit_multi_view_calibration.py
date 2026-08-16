"""Fit view-specific temperature and bias from validation candidates only."""

import argparse
import hashlib
import json
from pathlib import Path

from .multi_view_calibration import fit_calibration_from_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-views", type=int, required=True)
    parser.add_argument("--max-iter", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.validation_candidates)
    with open(source) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    calibration = fit_calibration_from_rows(
        rows, n_views=args.n_views, max_iter=args.max_iter
    )
    calibration["validation_candidates"] = str(source.resolve())
    calibration["validation_candidates_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(calibration, handle, indent=2)
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
