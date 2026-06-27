#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export HF_HOME="$ROOT_DIR/hf-cache"
export HUGGINGFACE_HUB_CACHE="$ROOT_DIR/hf-cache/hub"
export HF_DATASETS_CACHE="$ROOT_DIR/hf-cache/datasets"
export TRANSFORMERS_CACHE="$ROOT_DIR/hf-cache/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

uv venv "$ROOT_DIR/.venv-colqwen2" --python 3.12

echo "$ROOT_DIR/.venv-vidore/lib/python3.12/site-packages" \
  > "$ROOT_DIR/.venv-colqwen2/lib/python3.12/site-packages/vidore_baseline_packages.pth"

uv pip install --python "$ROOT_DIR/.venv-colqwen2/bin/python" \
  --upgrade \
  --no-deps \
  "transformers>=4.45,<4.50" \
  "colpali-engine==0.3.7"

uv pip install --python "$ROOT_DIR/.venv-colqwen2/bin/python" \
  --upgrade \
  --no-deps \
  "tokenizers>=0.20,<0.22" \
  "safetensors>=0.4" \
  "huggingface-hub>=0.26,<1.0" \
  "python-dotenv"
