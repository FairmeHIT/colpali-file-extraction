#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_zh_mineru_benchmark.py \
  --model-name vidore/colpali \
  --limit 20 \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1

