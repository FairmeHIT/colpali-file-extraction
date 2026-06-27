#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_zh_mineru_benchmark.py \
  --model-name google/siglip-so400m-patch14-384 \
  --limit 50 \
  --batch-query 8 \
  --batch-doc 8 \
  --batch-score 8

