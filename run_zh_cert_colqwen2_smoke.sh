#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$ROOT_DIR/.venv-colqwen2/bin/python" ]]; then
  echo "Missing .venv-colqwen2. Run: bash setup_colqwen2_env.sh" >&2
  exit 1
fi

source "$ROOT_DIR/env_colqwen2.sh"

python "$ROOT_DIR/scripts/evaluate_zh_cert_benchmark.py" \
  --backend colqwen2 \
  --model-name vidore/colqwen2-v0.1 \
  --classes businessLicense,ID \
  --n-per-class 1 \
  --n-others 2 \
  --seed 42 \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1
