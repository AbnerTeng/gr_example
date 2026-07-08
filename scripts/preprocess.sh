#!/bin/bash

set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ] && [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "Generate Document Embeddings"

python -m src.generate_embedding "$@"

echo "Building RQ docids"

python -m src.assign_rq_docids "$@"

echo "Generate Pseudo Queries with docTTTTTquery"

python -m src.gen_pseudo_queries "$@"

echo "Preprocess data"

python -m src.prep_data "$@"
