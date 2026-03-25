#!/bin/bash -l
#SBATCH --job-name=vbench_core_runs
#SBATCH --output=/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/slurm_logs/vbench_core_runs_%A_%a.out
#SBATCH --error=/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/slurm_logs/vbench_core_runs_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3%4

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver}"
COMBINED_SCRIPT="${COMBINED_SCRIPT:-${PROJECT_ROOT}/run_vbench_combined.sh}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

RUN_DIRS=(
  "/home/vault/v123be/v123be36/sceneweaver_runs/core_thirsty_crow_260319_225546"
  "/home/vault/v123be/v123be36/sceneweaver_runs/core_tortoise_and_hare_260319_225652"
  "/home/vault/v123be/v123be36/sceneweaver_runs/core_lion_and_mouse_260319_225545"
  "/home/vault/v123be/v123be36/sceneweaver_runs/core_fox_and_grapes_260319_225547"
)

if [ "${ARRAY_INDEX}" -lt 0 ] || [ "${ARRAY_INDEX}" -ge "${#RUN_DIRS[@]}" ]; then
  echo "Array index ${ARRAY_INDEX} is out of range for ${#RUN_DIRS[@]} run directories."
  exit 1
fi

TARGET_RUN_DIR="${RUN_DIRS[${ARRAY_INDEX}]}"
TARGET_NAME="$(basename "${TARGET_RUN_DIR}")"
REPORT_ROOT_DEFAULT="${TARGET_RUN_DIR}/benchmark_reports"

if [ ! -x "${COMBINED_SCRIPT}" ]; then
  echo "Combined benchmark launcher not found or not executable: ${COMBINED_SCRIPT}"
  exit 1
fi

if [ ! -d "${TARGET_RUN_DIR}" ]; then
  echo "Run directory does not exist: ${TARGET_RUN_DIR}"
  exit 1
fi

if [ ! -d "${TARGET_RUN_DIR}/clips" ]; then
  echo "Missing clips directory in run: ${TARGET_RUN_DIR}"
  exit 1
fi

if [ ! -f "${TARGET_RUN_DIR}/run_log.jsonl" ]; then
  echo "Missing run_log.jsonl in run: ${TARGET_RUN_DIR}"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs"
mkdir -p "${REPORT_ROOT_DEFAULT}"
cd "${PROJECT_ROOT}"

export EVAL_TARGET="${TARGET_RUN_DIR}"
export STORY_RUN_DIR="${TARGET_RUN_DIR}"
export STORY_SLUG="${STORY_SLUG:-${TARGET_NAME}}"
export RUN_NAME_BASE="${RUN_NAME_BASE:-${TARGET_NAME}}"
export REPORT_ROOT="${REPORT_ROOT:-${REPORT_ROOT_DEFAULT}}"
export ENV_PATH="${ENV_PATH:-sceneweaver_runtime}"
export CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
export INSTALL_VBENCH="${INSTALL_VBENCH:-1}"
export INSTALL_VIDEOBENCH="${INSTALL_VIDEOBENCH:-1}"
export VBENCH_PACKAGE="${VBENCH_PACKAGE:-vbench}"
export VIDEOBENCH_PACKAGE="${VIDEOBENCH_PACKAGE:-git+https://github.com/Video-Bench/Video-Bench.git}"
export VBENCH_EXTRA_PIP_PACKAGES="${VBENCH_EXTRA_PIP_PACKAGES:-setuptools==59.6.0 scikit-build imageio imageio-ffmpeg}"
export USE_PROXY="${USE_PROXY:-1}"
export PROXY_URL="${PROXY_URL:-http://proxy:80}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export RUN_CONTINUITY="${RUN_CONTINUITY:-1}"

if [ -n "${VIDEOBENCH_CONFIG_PATH:-}" ] && [ -f "${VIDEOBENCH_CONFIG_PATH}" ]; then
  export RUN_WINDOW_PROMPT="${RUN_WINDOW_PROMPT:-1}"
  export REQUIRE_WINDOW_PROMPT="${REQUIRE_WINDOW_PROMPT:-1}"
else
  export RUN_WINDOW_PROMPT="${RUN_WINDOW_PROMPT:-0}"
  export REQUIRE_WINDOW_PROMPT=0
  echo "VIDEOBENCH_CONFIG_PATH is not set or missing; running VBench continuity only."
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TARGET_RUN_DIR=${TARGET_RUN_DIR}"
echo "TARGET_NAME=${TARGET_NAME}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "EVAL_TARGET=${EVAL_TARGET}"
echo "RUN_WINDOW_PROMPT=${RUN_WINDOW_PROMPT}"
echo "RUN_CONTINUITY=${RUN_CONTINUITY}"
echo "VIDEOBENCH_CONFIG_PATH=${VIDEOBENCH_CONFIG_PATH:-}"

exec bash "${COMBINED_SCRIPT}"
