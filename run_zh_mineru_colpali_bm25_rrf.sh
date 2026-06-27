#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_zh_mineru_hybrid.py \
  --mode rrf \
  --model-name vidore/colpali \
  --rrf-k 60 \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1

