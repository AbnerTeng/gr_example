#!/bin/bash

set -euo pipefail

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python -m src.analyze_rq_neighbor_prefix "$@"
