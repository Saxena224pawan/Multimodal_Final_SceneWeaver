#!/bin/bash -l
#SBATCH --job-name=test_i2v_gen
#SBATCH --output=slurm_logs/test_i2v_gen_%j.out
#SBATCH --error=slurm_logs/test_i2v_gen_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=a1622
#SBATCH --cpus-per-task=4

set -euo pipefail

# Some cluster profile scripts assume this exists; keep nounset-safe default.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Defaults are portable; override with env vars if needed.
DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-sceneweaver311}"

CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
USE_MODULES="${USE_MODULES:-1}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12-conda}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1}"

# Optional HPC module setup.
if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge || true
  [ -n "${PYTHON_MODULE}" ] && module load "${PYTHON_MODULE}" || true
  [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

# If a non-base env is already active in the current shell, keep it by default.
# Ignore the generic "base" env so the launcher can fall back to the project runtime.
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ] && [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ]; then
  ENV_PATH="${CONDA_DEFAULT_ENV}"
fi

# Optional default env when neither ENV_PATH nor VENV_PATH is provided.
if [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ] && [ -n "${DEFAULT_ENV_PATH}" ]; then
  ENV_PATH="${DEFAULT_ENV_PATH}"
fi

# Optional runtime activation.
if [ -n "${ENV_PATH}" ]; then
  if [ -f "${CONDA_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  fi
  command -v conda >/dev/null 2>&1 || { echo "Conda command not found. Set CONDA_SH correctly or activate env before running."; exit 1; }
  conda activate "${ENV_PATH}"
elif [ -n "${VENV_PATH}" ]; then
  [ -f "${VENV_PATH}/bin/activate" ] || { echo "Virtualenv activate script not found at ${VENV_PATH}/bin/activate"; exit 1; }
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Generation parameters
MODEL_PATH="${MODEL_PATH:-}"
PROMPT="${PROMPT:-a car driving on a highway}"
OUTPUT_PATH="${OUTPUT_PATH:-test_video_base.mp4}"
NUM_FRAMES="${NUM_FRAMES:-16}"
STEPS="${STEPS:-20}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python/python3 found."
    exit 1
  fi
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PROMPT=${PROMPT}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "STEPS=${STEPS}"

# Run generation test
CMD=("${PYTHON_BIN}" test_generation.py
  --prompt "${PROMPT}"
  --output_path "${OUTPUT_PATH}"
  --num_frames "${NUM_FRAMES}"
  --steps "${STEPS}")

if [ -n "${MODEL_PATH}" ]; then
  CMD+=(--model_path "${MODEL_PATH}")
fi

"${CMD[@]}"