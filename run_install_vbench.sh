#!/bin/bash -l
#SBATCH --job-name=install_vbench
#SBATCH --output=slurm_logs/install_vbench_%j.out
#SBATCH --error=slurm_logs/install_vbench_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4

set -euo pipefail

COMMON_SLURM_ROOT="${COMMON_SLURM_ROOT:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}}"
COMMON_SLURM_SH="${COMMON_SLURM_ROOT}/slurm_common.sh"
# shellcheck source=./slurm_common.sh
source "${COMMON_SLURM_SH}"

# Keep nounset-safe default expected by some cluster profiles.
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

VBENCH_PACKAGE="${VBENCH_PACKAGE:-vbench}"
UPGRADE_PIP="${UPGRADE_PIP:-1}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-setuptools==80.9.0}"
DRY_RUN="${DRY_RUN:-0}"
USE_PROXY="${USE_PROXY:-1}"
PROXY_URL="${PROXY_URL:-${SCENEWEAVER_PROXY_URL}}"
NO_PROXY="${NO_PROXY:-${SCENEWEAVER_NO_PROXY}}"

if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge || true
  [ -n "${PYTHON_MODULE}" ] && module load "${PYTHON_MODULE}" || true
  [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs"
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

# Cluster tutorial proxy settings for package downloads on compute nodes.
if [ "${USE_PROXY}" = "1" ] && [ -n "${PROXY_URL}" ]; then
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  export no_proxy="${NO_PROXY}"
  export NO_PROXY="${NO_PROXY}"
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_PATH=${ENV_PATH:-}"
echo "VENV_PATH=${VENV_PATH:-}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "VBENCH_PACKAGE=${VBENCH_PACKAGE}"
echo "UPGRADE_PIP=${UPGRADE_PIP}"
echo "EXTRA_PIP_PACKAGES=${EXTRA_PIP_PACKAGES}"
echo "DRY_RUN=${DRY_RUN}"
echo "USE_PROXY=${USE_PROXY}"
echo "PROXY_URL=${PROXY_URL}"
echo "NO_PROXY=${NO_PROXY}"

bash -n "${BASH_SOURCE[0]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] ${PYTHON_BIN} -m pip install --upgrade ${VBENCH_PACKAGE} ${EXTRA_PIP_PACKAGES}"
  exit 0
fi

if [ "${UPGRADE_PIP}" = "1" ]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip
fi

"${PYTHON_BIN}" -m pip install --upgrade "${VBENCH_PACKAGE}"

if [ -n "${EXTRA_PIP_PACKAGES}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_PIP_PACKAGES})
  "${PYTHON_BIN}" -m pip install --upgrade "${EXTRA_ARR[@]}"
fi

if ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
  echo "pkg_resources missing after requested installs; forcing setuptools==80.9.0"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "setuptools==80.9.0"
fi

if ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
  echo "ERROR: pkg_resources still missing after fallback install."
  exit 1
fi

"${PYTHON_BIN}" -m pip show vbench || true
if command -v vbench >/dev/null 2>&1; then
  vbench --help >/dev/null 2>&1 || true
  echo "vbench executable found: $(command -v vbench)"
else
  echo "WARNING: vbench executable not found in PATH after install."
fi

echo "VBench installation step finished."
