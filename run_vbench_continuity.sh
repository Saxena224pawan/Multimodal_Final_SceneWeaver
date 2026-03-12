#!/bin/bash -l
#SBATCH --job-name=vbench_cont
#SBATCH --output=slurm_logs/vbench_cont_%j.out
#SBATCH --error=slurm_logs/vbench_cont_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

set -euo pipefail

# Keep nounset-safe default expected by some cluster profiles.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-sceneweaver_runtime}"

CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
USE_MODULES="${USE_MODULES:-0}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12-conda}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1}"
PYTHON_BIN="${PYTHON_BIN:-}"

VBENCH_BIN="${VBENCH_BIN:-vbench}"
VBENCH_PACKAGE="${VBENCH_PACKAGE:-vbench}"
INSTALL_VBENCH="${INSTALL_VBENCH:-1}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
VBENCH_EXTRA_PIP_PACKAGES="${VBENCH_EXTRA_PIP_PACKAGES:-setuptools==80.9.0}"
USE_PROXY="${USE_PROXY:-1}"
PROXY_URL="${PROXY_URL:-http://proxy:80}"
NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
MODE="${MODE:-custom_input}"
SEQUENCE_MODE="${SEQUENCE_MODE:-concat_windows}"
DIMENSIONS="${DIMENSIONS:-subject_consistency,background_consistency,motion_smoothness,temporal_flickering}"
SEQUENCE_FALLBACK_PIP_PACKAGES="${SEQUENCE_FALLBACK_PIP_PACKAGES:-imageio imageio-ffmpeg}"
NGPUS="${NGPUS:-1}"
VIDEOS_PATH="${VIDEOS_PATH:-}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs}"
REPORT_ROOT="${REPORT_ROOT:-${PROJECT_ROOT}/outputs/reports/vbench_continuity}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-vbench_continuity_${RUN_STAMP}}"
VBENCH_EXTRA_ARGS="${VBENCH_EXTRA_ARGS:-}"
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
# Required for vbench motion_smoothness with newer PyTorch defaults.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

# Cluster tutorial proxy settings for downloading packages on compute nodes.
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

if [ "${DRY_RUN}" = "0" ] && ! command -v "${VBENCH_BIN}" >/dev/null 2>&1; then
  if [ "${INSTALL_VBENCH}" = "1" ]; then
    echo "VBench executable not found: ${VBENCH_BIN}"
    echo "Attempting install with package: ${VBENCH_PACKAGE}"
    if [ "${UPGRADE_PIP}" = "1" ]; then
      "${PYTHON_BIN}" -m pip install --upgrade pip
    fi
    "${PYTHON_BIN}" -m pip install --upgrade "${VBENCH_PACKAGE}"
    if [ -n "${VBENCH_EXTRA_PIP_PACKAGES}" ]; then
      # shellcheck disable=SC2206
      EXTRA_ARR=(${VBENCH_EXTRA_PIP_PACKAGES})
      "${PYTHON_BIN}" -m pip install --upgrade "${EXTRA_ARR[@]}"
    fi
    hash -r
  fi
fi

if [ "${DRY_RUN}" = "0" ] && ! command -v "${VBENCH_BIN}" >/dev/null 2>&1; then
  echo "VBench executable not found after install attempt: ${VBENCH_BIN}"
  echo "Set VBENCH_BIN if custom executable name/path is used."
  exit 1
fi

# Some VBench dimensions import `pkg_resources` via `clip`.
if [ "${DRY_RUN}" = "0" ] && ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
  echo "pkg_resources missing; installing ${VBENCH_EXTRA_PIP_PACKAGES}"
  if [ -n "${VBENCH_EXTRA_PIP_PACKAGES}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${VBENCH_EXTRA_PIP_PACKAGES})
    "${PYTHON_BIN}" -m pip install --upgrade "${EXTRA_ARR[@]}"
  fi
fi

# setuptools>=82 can still miss pkg_resources in this environment; force a compatible version.
if [ "${DRY_RUN}" = "0" ] && ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
  echo "pkg_resources still missing; forcing setuptools==80.9.0"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "setuptools==80.9.0"
fi

if [ "${DRY_RUN}" = "0" ] && ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
  echo "ERROR: pkg_resources is still unavailable after fallback install."
  echo "Try: ${PYTHON_BIN} -m pip install --force-reinstall --no-deps setuptools==80.9.0"
  exit 1
fi

if [ "${DRY_RUN}" = "0" ] && [ "${SEQUENCE_MODE}" = "concat_windows" ] && ! command -v ffmpeg >/dev/null 2>&1; then
  if ! "${PYTHON_BIN}" -c "import imageio, imageio_ffmpeg" >/dev/null 2>&1; then
    echo "ffmpeg not found; installing Python concat fallback deps: ${SEQUENCE_FALLBACK_PIP_PACKAGES}"
    # shellcheck disable=SC2206
    SEQ_ARR=(${SEQUENCE_FALLBACK_PIP_PACKAGES})
    "${PYTHON_BIN}" -m pip install --upgrade "${SEQ_ARR[@]}"
  fi
fi

if [ "${DRY_RUN}" = "0" ] && [ "${SEQUENCE_MODE}" = "concat_windows" ]; then
  if ! command -v ffmpeg >/dev/null 2>&1 && ! "${PYTHON_BIN}" -c "import imageio, imageio_ffmpeg" >/dev/null 2>&1; then
    echo "ERROR: concat_windows requires either ffmpeg in PATH or Python packages imageio+imageio-ffmpeg."
    echo "Fallback: set SEQUENCE_MODE=per_clip."
    exit 1
  fi
fi

CMD=("${PYTHON_BIN}" scripts/09_eval_vbench_continuity.py
  --mode "${MODE}"
  --sequence_mode "${SEQUENCE_MODE}"
  --report_root "${REPORT_ROOT}"
  --run_name "${RUN_NAME}"
  --dimensions "${DIMENSIONS}"
  --vbench_bin "${VBENCH_BIN}"
  --ngpus "${NGPUS}"
)

if [ -n "${VIDEOS_PATH}" ]; then
  CMD+=(--videos_path "${VIDEOS_PATH}")
fi
if [ -n "${OUTPUTS_ROOT}" ]; then
  CMD+=(--outputs_root "${OUTPUTS_ROOT}")
fi
if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry_run)
fi

if [ -n "${VBENCH_EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${VBENCH_EXTRA_ARGS})
  for arg in "${EXTRA_ARR[@]}"; do
    CMD+=(--extra_arg "${arg}")
  done
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_PATH=${ENV_PATH:-}"
echo "VENV_PATH=${VENV_PATH:-}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "VBENCH_BIN=${VBENCH_BIN}"
echo "VBENCH_PACKAGE=${VBENCH_PACKAGE}"
echo "INSTALL_VBENCH=${INSTALL_VBENCH}"
echo "UPGRADE_PIP=${UPGRADE_PIP}"
echo "VBENCH_EXTRA_PIP_PACKAGES=${VBENCH_EXTRA_PIP_PACKAGES}"
echo "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}"
echo "USE_PROXY=${USE_PROXY}"
echo "PROXY_URL=${PROXY_URL}"
echo "NO_PROXY=${NO_PROXY}"
echo "MODE=${MODE}"
echo "SEQUENCE_MODE=${SEQUENCE_MODE}"
echo "DIMENSIONS=${DIMENSIONS}"
echo "SEQUENCE_FALLBACK_PIP_PACKAGES=${SEQUENCE_FALLBACK_PIP_PACKAGES}"
echo "VIDEOS_PATH=${VIDEOS_PATH:-<latest story run auto>}"
echo "OUTPUTS_ROOT=${OUTPUTS_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "NGPUS=${NGPUS}"
echo "DRY_RUN=${DRY_RUN}"
echo "VBENCH_EXTRA_ARGS=${VBENCH_EXTRA_ARGS}"

bash -n "${BASH_SOURCE[0]}"
"${CMD[@]}"
