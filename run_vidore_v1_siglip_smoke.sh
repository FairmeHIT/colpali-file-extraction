#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

vidore-benchmark evaluate-retriever \
  --model-name google/siglip-so400m-patch14-384 \
  --dataset-name vidore/docvqa_test_subsampled \
  --split test \
  --batch-query 8 \
  --batch-doc 8 \
  --batch-score 8
