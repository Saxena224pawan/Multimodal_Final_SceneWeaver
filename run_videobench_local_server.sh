#!/bin/bash -l
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-sceneweaver_runtime}"
PYTHON_BIN="${PYTHON_BIN:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/home/vault/v123be/v123be36/sceneweaver_models/Qwen2.5-VL-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-local-videobench}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
SERVER_EXTRA_PIP_PACKAGES="${SERVER_EXTRA_PIP_PACKAGES:-huggingface_hub[cli] starlette pillow}"
HF_CACHE_SNAPSHOT_ROOT="${HF_CACHE_SNAPSHOT_ROOT:-/home/hpc/v123be/v123be36/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python not found in env ${ENV_NAME}"
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" -c "import transformers, starlette, uvicorn, PIL, torch" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --upgrade ${SERVER_EXTRA_PIP_PACKAGES}
fi

model_dir_is_valid() {
  local dir_path="$1"
  "${PYTHON_BIN}" - "$dir_path" <<'PYVALID'
import glob
import json
import os
import sys

model_dir = sys.argv[1]
config_path = os.path.join(model_dir, "config.json")
if not os.path.isfile(config_path):
    raise SystemExit(1)
try:
    with open(config_path, "r", encoding="utf-8") as handle:
        json.load(handle)
except Exception:
    raise SystemExit(1)

weight_patterns = (
    os.path.join(model_dir, "*.safetensors"),
    os.path.join(model_dir, "pytorch_model*.bin"),
    os.path.join(model_dir, "*.bin"),
)
has_weights = any(glob.glob(pattern) for pattern in weight_patterns)
raise SystemExit(0 if has_weights else 1)
PYVALID
}

is_scratch_like_dir() {
  local dir_path="$1"
  case "${dir_path}" in
    /tmp/*|/var/tmp/*)
      return 0
      ;;
  esac
  if [ -n "${TMPDIR:-}" ] && [[ "${dir_path}" == "${TMPDIR%/}/"* ]]; then
    return 0
  fi
  if [ -n "${SLURM_TMPDIR:-}" ] && [[ "${dir_path}" == "${SLURM_TMPDIR%/}/"* ]]; then
    return 0
  fi
  return 1
}

if ! model_dir_is_valid "${MODEL_DIR}" && [ -d "${HF_CACHE_SNAPSHOT_ROOT}" ]; then
  cached_snapshot="$(find "${HF_CACHE_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [ -n "${cached_snapshot}" ] && model_dir_is_valid "${cached_snapshot}"; then
    MODEL_DIR="${cached_snapshot}"
  fi
fi

if ! model_dir_is_valid "${MODEL_DIR}"; then
  if [ "${DOWNLOAD_MODEL}" != "1" ]; then
    echo "Model not found at ${MODEL_DIR} and DOWNLOAD_MODEL=0"
    exit 1
  fi
  if ! is_scratch_like_dir "${MODEL_DIR}"; then
    scratch_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/videobench_models}}"
    MODEL_DIR="${scratch_root%/}/$(basename "${MODEL_DIR}")"
    echo "Falling back to scratch model dir: ${MODEL_DIR}"
  fi
  if [ -d "${MODEL_DIR}" ] && ! model_dir_is_valid "${MODEL_DIR}" && is_scratch_like_dir "${MODEL_DIR}"; then
    echo "Removing incomplete scratch model dir: ${MODEL_DIR}"
    rm -rf "${MODEL_DIR}"
  fi
  mkdir -p "${MODEL_DIR}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
  else
    "${PYTHON_BIN}" -m pip install --upgrade "huggingface_hub[cli]"
    hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
  fi
  if ! model_dir_is_valid "${MODEL_DIR}"; then
    echo "Downloaded model dir is still incomplete or invalid: ${MODEL_DIR}"
    exit 1
  fi
fi

echo "Serving local Video-Bench model"
echo "MODEL_DIR=${MODEL_DIR}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "BASE_URL=http://${HOST}:${PORT}/v1"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_local_openai_vlm_server.py" \
  --model-dir "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}"
