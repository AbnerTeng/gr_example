#!/bin/bash

set -euo pipefail

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python -m src.analyze_rq_semanticity "$@"

# bash scripts/analyze_rq_semanticity.sh   --target-pairs 10000   --output-dir outputs/rq_semanticity_10k