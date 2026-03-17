#!/bin/bash -l
#SBATCH --job-name=finetune_i2v
#SBATCH --output=slurm_logs/finetune_i2v_%j.out
#SBATCH --error=slurm_logs/finetune_i2v_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

set -euo pipefail

# Some cluster profile scripts assume this exists; keep nounset-safe default.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Defaults are portable; override with env vars if needed.
DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-sceneweaver311}"

# Optional HPC module setup.
if [ "${USE_MODULES:-0}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge || true
  [ -n "${PYTHON_MODULE:-python/3.12-conda}" ] && module load "${PYTHON_MODULE:-python/3.12-conda}" || true
  [ -n "${CUDA_MODULE:-cuda/12.4.1}" ] && module load "${CUDA_MODULE:-cuda/12.4.1}" || true
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
  if [ -f "${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
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

# Fine-tuning parameters
DATASET="${DATASET:-cifar10}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/finetune_i2v_$(date +%Y%m%d_%H%M%S)}"

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
echo "DATASET=${DATASET}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# Run the fine-tuning
"${PYTHON_BIN}" scripts/finetune_i2v_model.py \
  --dataset "${DATASET}" \
  --max_samples "${MAX_SAMPLES}" \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --output_dir "${OUTPUT_DIR}"