#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_zh_mineru_hybrid.py \
  --mode linear \
  --model-name vidore/colpali \
  --alpha 0.5 \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1

