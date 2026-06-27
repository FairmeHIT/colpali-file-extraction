#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VIRTUAL_ENV="$ROOT_DIR/.venv-colqwen2"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export HF_HOME="$ROOT_DIR/hf-cache"
export HUGGINGFACE_HUB_CACHE="$ROOT_DIR/hf-cache/hub"
export HF_DATASETS_CACHE="$ROOT_DIR/hf-cache/datasets"
export TRANSFORMERS_CACHE="$ROOT_DIR/hf-cache/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::SyntaxWarning}"
