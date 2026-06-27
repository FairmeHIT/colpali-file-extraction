# Chinese Certificate Benchmark

This workspace is isolated from the official `colpali/`, `vidore-benchmark/`, and the existing `local_data/zh_corpus/` benchmark.

## Data Layout

All first-level directories except `others/` are treated as certificate classes. `others/` is used as the negative/background pool.

Current class directories:

```text
CESSCN_design_inte
CESSCN_emergency_resp
CESSCN_risk_eval
CESSCN_safety_train
ID
ISO14000
ISO45001
ISO9000
SA8000
academicCertificate
businessLicense
degreeCertificate
legalLicense
others
```

## Evaluation

The local evaluator randomly samples `N` images from each certificate class, mixes them with images sampled from `others/`, and evaluates ColPali retrieval.

Example queries:

```text
查找一张营业执照图片。
查找一张身份证图片。
查找一张ISO9000质量管理体系认证证书图片。
```

For each class query, all `N` sampled images from that class are treated as positives. `others/` rows keep `query: null` and act as negatives.

Validate dataset construction without loading a model:

```bash
source env.sh
python scripts/evaluate_zh_cert_benchmark.py \
  --n-per-class 3 \
  --n-others 30 \
  --seed 42 \
  --dry-run-build
```

Run ColPali:

```bash
bash run_zh_cert_colpali_smoke.sh
```

Run ColQwen2 in an isolated environment:

```bash
bash setup_colqwen2_env.sh
bash run_zh_cert_colqwen2_smoke.sh
```

ColQwen2 is intentionally kept outside `.venv-vidore` because the current
baseline environment uses `transformers==4.41.2`, while `vidore/colqwen2-v0.1`
requires Qwen2-VL support from newer transformers. Do not upgrade `.venv-vidore`
if you want to keep existing ColPali/ViDoRe baselines reproducible.

For a direct horizontal comparison, use the same `--classes`, `--n-per-class`,
`--n-others`, and `--seed` for both models, changing only:

```bash
--backend registry --model-name vidore/colpali
--backend colqwen2 --model-name vidore/colqwen2-v0.1
```

Or specify classes manually:

```bash
source env.sh
python scripts/evaluate_zh_cert_benchmark.py \
  --classes businessLicense,ID \
  --n-per-class 3 \
  --n-others 30 \
  --seed 42 \
  --model-name vidore/colpali \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1
```

## Outputs

Each run writes an isolated sampled dataset and manifest under:

```text
local_data/zh_cert_benchmark/benchmark_runs/seed_<seed>_n_<N>_others_<M>/
```

Files:

| File | Description |
|---|---|
| `vidore_compat_test.jsonl` | Sampled local dataset with class queries on positive rows and `query: null` on `others` rows |
| `manifest.json` | Classes, selected image paths, and counts |
| `registry_vidore_colpali_metrics.json` | Metrics after running ColPali |
| `colqwen2_vidore_colqwen2-v0.1_metrics.json` | Metrics after running ColQwen2 |
