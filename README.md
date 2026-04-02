# Example GR

The standard training recipe to train a Generative Retrieval (GR) model

## Quick start

Use `uv` to build the virtual environment

```bash
uv sync
```

The default version of PyTorch is companion with cuda version 12.8, if your cuda version is different from it, please refer to the official website of PyTorch and manually install `torch` package

## Prepare dataset

> We use MSMARCO-100k (microsoft/ms_marco) as the example dataset. Note that it's the v1.1 one.

The overall data preparation script can be directly run by

```bash
bash scripts/preprocess.sh
```

Below is the detail of data preparation and preprocessing

### Assign RQ docids

- We assign unique docid to each document with Residual Quantization (RQ) codebooks. 
- We first utilize a light weight embedding model (Qwen3-0.6B-Embedding) to generate document embeddings (The code is in `src/generate_embedding.py`).

### Generate pseudo queries

For reaching better retrieval performance, we generate 5 additional pseudo queries for each document using the docTTTTTquery [Noegueira et al, 2019]

## Model Training

We adopt T5-Large as the base GR model, which aligns with most GR research.

Execute the below script to start training the model

```bash
bash script/train.sh
```

All model and traning configurations are stored in `configs/train.yaml`

## Evaluation performance

| Model/Metrics | Hits@1 | Hits@5 | Hits@10 |
|---------------|--------|--------|---------|
| T5-Large      | 0.2544 | 0.4553 | 0.5089  |
