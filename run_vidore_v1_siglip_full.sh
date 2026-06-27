#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_vidore_v1.py \
  --model-name google/siglip-so400m-patch14-384 \
  --split test \
  --batch-query "${BATCH_QUERY:-8}" \
  --batch-doc "${BATCH_DOC:-8}" \
  --batch-score "${BATCH_SCORE:-8}"
