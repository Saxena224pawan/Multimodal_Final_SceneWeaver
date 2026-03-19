#!/bin/bash -l
#SBATCH --job-name=videobench_wp
#SBATCH --output=slurm_logs/videobench_wp_%j.out
#SBATCH --error=slurm_logs/videobench_wp_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

set -euo pipefail

COMMON_SLURM_ROOT="${COMMON_SLURM_ROOT:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}}"
COMMON_SLURM_SH="${COMMON_SLURM_ROOT}/slurm_common.sh"
# shellcheck source=./slurm_common.sh
source "${COMMON_SLURM_SH}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-${SCENEWEAVER_DEFAULT_ENV}}"

CONDA_SH="${CONDA_SH:-${SCENEWEAVER_CONDA_SH}}"
USE_MODULES="${USE_MODULES:-0}"
PYTHON_MODULE="${PYTHON_MODULE:-${SCENEWEAVER_PYTHON_MODULE}}"
CUDA_MODULE="${CUDA_MODULE:-${SCENEWEAVER_CUDA_MODULE}}"
PYTHON_BIN="${PYTHON_BIN:-}"

VIDEOBENCH_BIN="${VIDEOBENCH_BIN:-videobench}"
VIDEOBENCH_PACKAGE="${VIDEOBENCH_PACKAGE:-videobench}"
INSTALL_VIDEOBENCH="${INSTALL_VIDEOBENCH:-1}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
VIDEOBENCH_EXTRA_PIP_PACKAGES="${VIDEOBENCH_EXTRA_PIP_PACKAGES:-}"
USE_PROXY="${USE_PROXY:-1}"
PROXY_URL="${PROXY_URL:-${SCENEWEAVER_PROXY_URL}}"
NO_PROXY="${NO_PROXY:-${SCENEWEAVER_NO_PROXY}}"
VIDEOBENCH_CONFIG_PATH="${VIDEOBENCH_CONFIG_PATH:-}"
DIMENSIONS="${DIMENSIONS:-video-text consistency,action,scene,object_class,color}"
PROMPT_SOURCE="${PROMPT_SOURCE:-auto}"
MODE="${MODE:-auto}"
MODEL_NAME="${MODEL_NAME:-}"
LINK_MODE="${LINK_MODE:-auto}"
VIDEOS_PATH="${VIDEOS_PATH:-}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs}"
REPORT_ROOT="${REPORT_ROOT:-${PROJECT_ROOT}/outputs/reports/videobench_window_prompt}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-videobench_window_prompt_${RUN_STAMP}}"
VIDEOBENCH_EXTRA_ARGS="${VIDEOBENCH_EXTRA_ARGS:-}"
SKIP_MISSING_PROMPTS="${SKIP_MISSING_PROMPTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge || true
  [ -n "${PYTHON_MODULE}" ] && module load "${PYTHON_MODULE}" || true
  [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

if [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ]; then
  ENV_PATH="${DEFAULT_ENV_PATH}"
fi

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

if [ "${USE_PROXY}" = "1" ] && [ -n "${PROXY_URL}" ]; then
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  export no_proxy="${NO_PROXY}"
  export NO_PROXY="${NO_PROXY}"
fi

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python/python3 found."
    exit 1
  fi
fi

if [ "${DRY_RUN}" = "0" ] && ! command -v "${VIDEOBENCH_BIN}" >/dev/null 2>&1; then
  if [ "${INSTALL_VIDEOBENCH}" = "1" ]; then
    echo "Video-Bench executable not found: ${VIDEOBENCH_BIN}"
    echo "Attempting install with package: ${VIDEOBENCH_PACKAGE}"
    if [ "${UPGRADE_PIP}" = "1" ]; then
      "${PYTHON_BIN}" -m pip install --upgrade pip
    fi
    "${PYTHON_BIN}" -m pip install --upgrade "${VIDEOBENCH_PACKAGE}"
    if [ -n "${VIDEOBENCH_EXTRA_PIP_PACKAGES}" ]; then
      # shellcheck disable=SC2206
      EXTRA_PKG_ARR=(${VIDEOBENCH_EXTRA_PIP_PACKAGES})
      "${PYTHON_BIN}" -m pip install --upgrade "${EXTRA_PKG_ARR[@]}"
    fi
    hash -r
  fi
fi

if [ "${DRY_RUN}" = "0" ] && ! command -v "${VIDEOBENCH_BIN}" >/dev/null 2>&1; then
  echo "Video-Bench executable not found after install attempt: ${VIDEOBENCH_BIN}"
  echo "Set VIDEOBENCH_BIN if a custom executable name/path is used."
  exit 1
fi

if [ "${DRY_RUN}" = "0" ] && [ -z "${VIDEOBENCH_CONFIG_PATH}" ]; then
  echo "VIDEOBENCH_CONFIG_PATH is required for non-dry-run jobs."
  echo "Point it to the Video-Bench config.json containing your API credentials."
  exit 1
fi

if [ "${DRY_RUN}" = "0" ] && [ ! -f "${VIDEOBENCH_CONFIG_PATH}" ]; then
  echo "Video-Bench config path does not exist: ${VIDEOBENCH_CONFIG_PATH}"
  exit 1
fi

CMD=("${PYTHON_BIN}" scripts/10_eval_videobench_window_prompt.py
  --report_root "${REPORT_ROOT}"
  --run_name "${RUN_NAME}"
  --dimensions "${DIMENSIONS}"
  --prompt_source "${PROMPT_SOURCE}"
  --mode "${MODE}"
  --model_name "${MODEL_NAME}"
  --link_mode "${LINK_MODE}"
  --videobench_bin "${VIDEOBENCH_BIN}"
)

if [ -n "${VIDEOS_PATH}" ]; then
  CMD+=(--videos_path "${VIDEOS_PATH}")
fi
if [ -n "${OUTPUTS_ROOT}" ]; then
  CMD+=(--outputs_root "${OUTPUTS_ROOT}")
fi
if [ -n "${VIDEOBENCH_CONFIG_PATH}" ]; then
  CMD+=(--config_path "${VIDEOBENCH_CONFIG_PATH}")
fi
if [ "${SKIP_MISSING_PROMPTS}" = "1" ]; then
  CMD+=(--skip_missing_prompts)
fi
if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry_run)
fi

if [ -n "${VIDEOBENCH_EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARG_ARR=(${VIDEOBENCH_EXTRA_ARGS})
  for arg in "${EXTRA_ARG_ARR[@]}"; do
    CMD+=(--extra_arg "${arg}")
  done
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_PATH=${ENV_PATH:-}"
echo "VENV_PATH=${VENV_PATH:-}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "VIDEOBENCH_BIN=${VIDEOBENCH_BIN}"
echo "VIDEOBENCH_PACKAGE=${VIDEOBENCH_PACKAGE}"
echo "INSTALL_VIDEOBENCH=${INSTALL_VIDEOBENCH}"
echo "UPGRADE_PIP=${UPGRADE_PIP}"
echo "VIDEOBENCH_EXTRA_PIP_PACKAGES=${VIDEOBENCH_EXTRA_PIP_PACKAGES}"
echo "VIDEOBENCH_CONFIG_PATH=${VIDEOBENCH_CONFIG_PATH:-}"
echo "USE_PROXY=${USE_PROXY}"
echo "PROXY_URL=${PROXY_URL}"
echo "NO_PROXY=${NO_PROXY}"
echo "DIMENSIONS=${DIMENSIONS}"
echo "PROMPT_SOURCE=${PROMPT_SOURCE}"
echo "MODE=${MODE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "LINK_MODE=${LINK_MODE}"
echo "VIDEOS_PATH=${VIDEOS_PATH:-<latest story run auto>}"
echo "OUTPUTS_ROOT=${OUTPUTS_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "SKIP_MISSING_PROMPTS=${SKIP_MISSING_PROMPTS}"
echo "DRY_RUN=${DRY_RUN}"
printf 'COMMAND='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
