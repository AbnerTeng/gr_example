# Shared-T5 Multi-DocID Generative Retrieval

This repository implements Multi-DocID generative retrieval with one shared T5:

- each document receives nine residual-quantization codes;
- the codes are split into exactly three contiguous views of three codes;
- each source is expanded into three view-conditioned training examples;
- inference runs the same T5 under three view-specific trie constraints;
- route candidates are expanded through collision-preserving posting lists and fused at document level.

The primary parameter-free ranker is beam-conditional posterior fusion. Majority voting, validation-only calibration, and a lightweight linear reranker are retained as baselines or ablations.

## Environment

```bash
source .venv/bin/activate
export PYTHONPATH=.
```

The training and evaluation code expects PyTorch, Transformers, Hydra/OmegaConf, NumPy, and FAISS. W&B is optional for RQ construction and enabled for the formal T5 configs.

## Multi-DocID contract

For a nine-level RQ identifier, the three routes are serialized as:

```text
<view_0> source -> <r0_x> <r1_y> <r2_z>
<view_1> source -> <r3_x> <r4_y> <r5_z>
<view_2> source -> <r6_x> <r7_y> <r8_z>
```

Training, preparation, and standard evaluation require exactly three views. `--single-view` in `src.eval_multi_view_gr` is diagnostic inference only and does not change the artifact contract.

## Pipeline

### 1. Prepare corpus embeddings

NQ:

```bash
python -m src.prep_nq --help
```

Fixed MS MARCO 300K subset:

```bash
python -m src.build_msmarco_subset --help
python -m src.embed_msmarco300k --help
```

### 2. Construct nine-code RQ identifiers

```bash
python -m src.train_rq --config-name rq9_sliced
python -m src.train_rq --config-name msmarco300k_rq9_sliced
```

Both configs produce `rq_codes.npy`, `idx_to_rqid.json`, `view_codes.npy`, and `idx_to_view_ids.json`. The required view shape is `(documents, 3, 3)`.

### 3. Generate filtered pseudo queries

```bash
python -m src.gen_pseudo_queries_nq --help
python -m src.gen_pseudo_queries_msmarco300k --help
```

The MS MARCO generator produces exactly five normalized, unique, non-dev pseudo queries per retained document and validates contiguous resume output.

### 4. Prepare shared-T5 datasets

```bash
python -m src.prep_multi_view_gr --help
```

This produces exact-three-view training examples, validation/test query files, document/view mappings, and posting-compatible route artifacts.

### 5. Train the shared T5

```bash
python -m src.train --config-name train_multi_view_nq
python -m src.train --config-name train_multi_view_msmarco300k
python -m src.train --config-name train_multi_view_msmarco300k_docT5query5
```

The formal configs use micro-batch 16, gradient accumulation 8, and effective batch size 128.

### 6. Evaluate and fuse document candidates

```bash
python -m src.eval_multi_view_gr --help
python -m src.fit_multi_view_calibration --help
python -m src.fit_lightweight_multi_view_reranker --help
python -m src.eval_lightweight_multi_view_reranker --help
```

`src.eval_multi_view_gr` supports majority voting and beam-conditional posterior fusion. Route collisions retain all posting-list documents and split route mass uniformly across the collision set.

Single-DocID evaluation remains available as a controlled baseline:

```bash
python -m src.eval_gr --help
```

## Diagnostics

```bash
python -m src.eval_multi_view_codes --help
python -m src.eval_rq_view_predictability --help
```

These report route utilization/collisions and cross-view query predictability/complementarity.

## Tests

The test modules are self-contained and can be run without pytest:

```bash
for test_file in tests/test_*.py; do
  python "$test_file"
done
```

Generated data, checkpoints, outputs, logs, W&B runs, local research artifacts, and Python caches are excluded from Git.
