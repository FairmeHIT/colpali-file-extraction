#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python scripts/evaluate_zh_mineru_hybrid.py \
  --mode bm25

