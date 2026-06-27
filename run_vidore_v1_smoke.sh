#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

vidore-benchmark evaluate-retriever \
  --model-name vidore/colpali \
  --dataset-name vidore/docvqa_test_subsampled \
  --split test \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1
