#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_zh_cert_benchmark.py \
  --n-per-class 3 \
  --n-others 30 \
  --seed 42 \
  --model-name vidore/colpali \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1

