#!/bin/bash -l
#SBATCH --job-name=sceneweaver_full
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
#SBATCH --time=1:50:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16

set -euo pipefail

# Keep nounset-safe default for cluster profiles.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-sceneweaver311}"
CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"

USE_MODULES="${USE_MODULES:-0}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12-conda}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf}"
USE_OFFLINE_MODE="${USE_OFFLINE_MODE:-1}"

# Optional model download switches.
DOWNLOAD_VIDEO_MODEL="${DOWNLOAD_VIDEO_MODEL:-0}"
VIDEO_MODEL_REPO="${VIDEO_MODEL_REPO:-cerspense/zeroscope_v2_576w}"
VIDEO_MODEL_DIR="${VIDEO_MODEL_DIR:-${PROJECT_ROOT}/VIDEO_GENERATIVE_BACKBONE/$(basename "${VIDEO_MODEL_REPO}")}"

DOWNLOAD_DINO_MODEL="${DOWNLOAD_DINO_MODEL:-0}"
DINO_MODEL_REPO="${DINO_MODEL_REPO:-facebook/dinov2-small}"
DINO_MODEL_DIR="${DINO_MODEL_DIR:-${PROJECT_ROOT}/Globa_Local_Emb_Feedback/dinov2-small}"

# Pipeline controls.
VIDEO_MODE="${VIDEO_MODE:-dry-run}" # dry-run|command
VIDEO_COMMAND_TEMPLATE="${VIDEO_COMMAND_TEMPLATE:-}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-${DINO_MODEL_DIR}}"
FAIL_ON_MISSING_FRAMES="${FAIL_ON_MISSING_FRAMES:-0}"

SAMPLE_FRAMES="${SAMPLE_FRAMES:-8}"
LOCAL_GRID="${LOCAL_GRID:-3}"
GLOBAL_MIN_SIM="${GLOBAL_MIN_SIM:-0.78}"
LOCAL_MIN_SIM="${LOCAL_MIN_SIM:-0.72}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-24}"
VIDEO_FPS_OUT="${VIDEO_FPS_OUT:-8}"
VIDEO_STEPS="${VIDEO_STEPS:-20}"
VIDEO_GUIDANCE_SCALE="${VIDEO_GUIDANCE_SCALE:-6.5}"
VIDEO_DEVICE="${VIDEO_DEVICE:-cuda}"
VIDEO_DTYPE="${VIDEO_DTYPE:-float16}"

# Optional HPC module setup.
if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge || true
  [ -n "${PYTHON_MODULE}" ] && module load "${PYTHON_MODULE}" || true
  [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/.hf"
cd "${PROJECT_ROOT}"

# If env already active, keep it unless overridden.
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ]; then
  ENV_PATH="${CONDA_DEFAULT_ENV}"
fi
if [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ] && [ -n "${DEFAULT_ENV_PATH}" ]; then
  ENV_PATH="${DEFAULT_ENV_PATH}"
fi

if [ -n "${ENV_PATH}" ]; then
  if [ -f "${CONDA_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  fi
  command -v conda >/dev/null 2>&1 || {
    echo "Conda command not found. Set CONDA_SH correctly or activate env before running."
    exit 1
  }
  conda activate "${ENV_PATH}"
elif [ -n "${VENV_PATH}" ]; then
  [ -f "${VENV_PATH}/bin/activate" ] || {
    echo "Virtualenv activate script not found at ${VENV_PATH}/bin/activate"
    exit 1
  }
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME

if [ "${USE_OFFLINE_MODE}" = "1" ] && { [ "${DOWNLOAD_VIDEO_MODEL}" = "1" ] || [ "${DOWNLOAD_DINO_MODEL}" = "1" ]; }; then
  echo "USE_OFFLINE_MODE=1 conflicts with model download. Switching to online mode for this run."
  USE_OFFLINE_MODE="0"
fi

if [ "${USE_OFFLINE_MODE}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DIFFUSERS_OFFLINE=1
fi

if [ "${DOWNLOAD_VIDEO_MODEL}" = "1" ] || [ "${DOWNLOAD_DINO_MODEL}" = "1" ]; then
  HF_DL=""
  if command -v hf >/dev/null 2>&1; then
    HF_DL="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DL="huggingface-cli"
  else
    echo "No HF CLI found. Install huggingface_hub[cli] or pre-download model directories."
    exit 1
  fi
fi

if [ "${DOWNLOAD_VIDEO_MODEL}" = "1" ]; then
  mkdir -p "${VIDEO_MODEL_DIR}"
  "${HF_DL}" download "${VIDEO_MODEL_REPO}" --local-dir "${VIDEO_MODEL_DIR}"
fi

if [ "${DOWNLOAD_DINO_MODEL}" = "1" ]; then
  mkdir -p "${DINO_MODEL_DIR}"
  "${HF_DL}" download "${DINO_MODEL_REPO}" --local-dir "${DINO_MODEL_DIR}"
fi

if [ "${VIDEO_MODE}" = "command" ] && [ -z "${VIDEO_COMMAND_TEMPLATE}" ]; then
  VIDEO_COMMAND_TEMPLATE="python3 ${PROJECT_ROOT}/scripts/04_video/02_generate_window_with_diffusers.py --prompt-file {q_prompt_file} --out {q_output_video} --frames-dir {q_frames_dir} --model-id {q_model_path} --fallback-model-id {q_model_repo_id} --window-id {q_window_id} --num-frames ${VIDEO_NUM_FRAMES} --fps-out ${VIDEO_FPS_OUT} --steps ${VIDEO_STEPS} --guidance-scale ${VIDEO_GUIDANCE_SCALE} --device ${VIDEO_DEVICE} --dtype ${VIDEO_DTYPE}"
fi

RUNNER="${PROJECT_ROOT}/run_story_pipeline.sh"
[ -x "${RUNNER}" ] || chmod +x "${RUNNER}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_PATH=${ENV_PATH:-<none>}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "VIDEO_MODE=${VIDEO_MODE}"
echo "VIDEO_NUM_FRAMES=${VIDEO_NUM_FRAMES}"
echo "VIDEO_FPS_OUT=${VIDEO_FPS_OUT}"
echo "VIDEO_STEPS=${VIDEO_STEPS}"
echo "VIDEO_GUIDANCE_SCALE=${VIDEO_GUIDANCE_SCALE}"
echo "VIDEO_DEVICE=${VIDEO_DEVICE}"
echo "VIDEO_DTYPE=${VIDEO_DTYPE}"
echo "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}"
echo "USE_OFFLINE_MODE=${USE_OFFLINE_MODE}"
echo "SAMPLE_FRAMES=${SAMPLE_FRAMES}"
echo "LOCAL_GRID=${LOCAL_GRID}"
echo "GLOBAL_MIN_SIM=${GLOBAL_MIN_SIM}"
echo "LOCAL_MIN_SIM=${LOCAL_MIN_SIM}"
echo "EMBEDDING_DEVICE=${EMBEDDING_DEVICE}"

PIPELINE_CMD=(
  "${RUNNER}"
  --video-mode "${VIDEO_MODE}"
)

if [ "${VIDEO_MODE}" = "command" ]; then
  PIPELINE_CMD+=(--command-template "${VIDEO_COMMAND_TEMPLATE}")
fi
if [ -n "${EMBEDDING_MODEL_ID}" ]; then
  PIPELINE_CMD+=(--embedding-model-id "${EMBEDDING_MODEL_ID}")
fi
if [ "${FAIL_ON_MISSING_FRAMES}" = "1" ]; then
  PIPELINE_CMD+=(--fail-on-missing-frames)
fi

export PYTHON_BIN
export SAMPLE_FRAMES
export LOCAL_GRID
export GLOBAL_MIN_SIM
export LOCAL_MIN_SIM
export EMBEDDING_DEVICE

"${PIPELINE_CMD[@]}"

echo "[OK] SLURM pipeline finished."
