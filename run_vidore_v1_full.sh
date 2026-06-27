#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_vidore_v1.py \
  --model-name vidore/colpali \
  --split test \
  --batch-query "${BATCH_QUERY:-4}" \
  --batch-doc "${BATCH_DOC:-4}" \
  --batch-score "${BATCH_SCORE:-4}"
