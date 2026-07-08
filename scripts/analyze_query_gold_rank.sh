#!/bin/bash

set -euo pipefail

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python -m src.analyze_query_gold_rank "$@"
