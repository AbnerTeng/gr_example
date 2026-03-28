#!/bin/bash

source .venv/bin/activate

echo "Building RQ docids"

python -m src.assign_rq_docids

echo "Generate Pseudo Queries with docTTTTTquery"

python -m src.gen_pseudo_queries

echo "Preprocess data"

python -m src.prepare_data