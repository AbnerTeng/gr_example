# Example GR

The standard training recipe to train a Generative Retrieval (GR) model

## Quick start

Use `uv` to build the virtual environment

```bash
uv sync
```

The default PyTorch source is configured for CUDA 12.1 (`pytorch-cu121`). If your CUDA setup is different, adjust the `torch` source/package and refresh `uv.lock`.

## Prepare dataset

> We use MSMARCO-100k (microsoft/ms_marco) as the example dataset. Note that it's the v1.1 one.

The overall data preparation script can be directly run by

```bash
bash scripts/preprocess.sh
```

Below is the detail of data preparation and preprocessing

### Assign RQ docids

- We assign unique docid to each document with Residual Quantization (RQ) codebooks.
- We first use a lightweight embedding model (Qwen3-0.6B-Embedding) to generate document embeddings (the code is in `src/generate_embedding.py`).

### Generate pseudo queries

For better retrieval performance, we generate 5 additional pseudo queries for each document using docTTTTTquery [Nogueira et al, 2019].

## Model Training

We adopt T5-Large as the base GR model, which aligns with most GR research.

Execute the below script to start training the model

```bash
bash scripts/train.sh
```

All model and training configurations are stored in `configs/train.yaml`

## Evaluation performance

| Model/Metrics | Hits@1 | Hits@5 | Hits@10 |
|---------------|--------|--------|---------|
| T5-Large      | 0.2544 | 0.4553 | 0.5089  |

## Analysis utilities

After preprocessing, the repository includes several lightweight analysis scripts
for checking whether embedding similarity and RQ DocID prefixes are aligned. They
use `data/doc_embeddings.npy`, `data/rq_codes.npy` or `data/idx_to_rqid.json`, and
write CSV/JSON summaries plus optional plots under `outputs/`.

Query-side checks:

```bash
# Where does each query's gold document rank by embedding similarity?
bash scripts/analyze_query_gold_rank.sh --output-dir outputs/query_gold_rank

# Do retrieved top-K docs share RQ prefixes with the gold document?
bash scripts/analyze_query_rq_prefix.sh \
  --k-values 10,50,100 \
  --output-dir outputs/query_rq_prefix

# Is query-doc similarity predictive of sharing the gold RQ prefix?
bash scripts/analyze_query_rq_ranking_consistency.sh \
  --knn-k 100 \
  --output-dir outputs/query_rq_ranking_consistency
```

Document-side checks:

```bash
# Do documents with longer shared RQ prefixes have higher embedding similarity?
bash scripts/analyze_rq_semanticity.sh \
  --target-pairs 100000 \
  --output-dir outputs/rq_semanticity

# Do embedding nearest neighbors share RQ prefixes?
bash scripts/analyze_rq_neighbor_prefix.sh \
  --k-values 10,50,100 \
  --output-dir outputs/rq_neighbor_prefix

# Is doc-doc embedding similarity predictive of sharing RQ prefixes?
bash scripts/analyze_rq_ranking_consistency.sh \
  --knn-k 100 \
  --output-dir outputs/rq_ranking_consistency
```

Useful options shared by the analysis scripts include `--levels` for RQ prefix
depth, `--no-plots` to skip figures, and `--max-queries` / `--anchor-docs` for
quick smaller runs. Query-side scripts cache query embeddings at
`data/test_query_embeddings.npy` unless `--no-encode` is set.
