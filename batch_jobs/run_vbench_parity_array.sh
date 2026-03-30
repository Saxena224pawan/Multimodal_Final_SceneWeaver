#!/bin/bash -l
#SBATCH --job-name=vbench_parity
#SBATCH --output=slurm_logs/vbench_parity_%A_%a.out
#SBATCH --error=slurm_logs/vbench_parity_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3%4

set -euo pipefail

project_root="${PROJECT_ROOT:-/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver}"
combined_script="${COMBINED_SCRIPT:-${project_root}/run_vbench_combined.sh}"
array_index="${SLURM_ARRAY_TASK_ID:-0}"

story_keys=(
  "fox_and_grapes"
  "lion_and_mouse"
  "thirsty_crow"
  "tortoise_and_hare"
)

if [ "${array_index}" -lt 0 ] || [ "${array_index}" -ge "${#story_keys[@]}" ]; then
  echo "Array index ${array_index} is out of range for ${#story_keys[@]} stories."
  exit 1
fi

mode="${MODE:-core}"
run_tag="${RUN_TAG:-}"
[ -n "${run_tag}" ] || { echo "ERROR: RUN_TAG must be provided."; exit 1; }

hpc_vault_root="${HPC_VAULT_ROOT:-/home/vault/v123be/v123be36}"
sceneweaver_runs_root="${SCENEWEAVER_RUNS_ROOT:-${hpc_vault_root}/sceneweaver_runs}"
parity_runs_root="${PARITY_RUNS_ROOT:-${sceneweaver_runs_root}/screenweaver_runs_parity}"

story_key="${story_keys[$array_index]}"
target_run_dir="${parity_runs_root}/${mode}_${story_key}_${run_tag}"
target_name="$(basename "${target_run_dir}")"
report_root_default="${target_run_dir}/benchmark_reports"

[ -x "${combined_script}" ] || { echo "Combined benchmark launcher not found or not executable: ${combined_script}"; exit 1; }
[ -d "${target_run_dir}" ] || { echo "Run directory does not exist: ${target_run_dir}"; exit 1; }
[ -d "${target_run_dir}/clips" ] || { echo "Missing clips directory in run: ${target_run_dir}"; exit 1; }

mkdir -p "${project_root}/slurm_logs"
mkdir -p "${report_root_default}"
cd "${project_root}"

export EVAL_TARGET="${target_run_dir}"
export STORY_RUN_DIR="${target_run_dir}"
export STORY_SLUG="${STORY_SLUG:-${target_name}}"
export RUN_NAME_BASE="${RUN_NAME_BASE:-${target_name}}"
export REPORT_ROOT="${REPORT_ROOT:-${report_root_default}}"
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
export RUN_WINDOW_PROMPT="${RUN_WINDOW_PROMPT:-1}"
export REQUIRE_WINDOW_PROMPT="${REQUIRE_WINDOW_PROMPT:-1}"
export VIDEOBENCH_CONFIG_PATH="${VIDEOBENCH_CONFIG_PATH:-${project_root}/configs/videobench_config.json}"
export SEQUENCE_MODE="${SEQUENCE_MODE:-concat_windows}"

[ -f "${VIDEOBENCH_CONFIG_PATH}" ] || { echo "Video-Bench config missing at ${VIDEOBENCH_CONFIG_PATH}"; exit 1; }

echo "PROJECT_ROOT=${project_root}"
echo "TARGET_RUN_DIR=${target_run_dir}"
echo "TARGET_NAME=${target_name}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "EVAL_TARGET=${EVAL_TARGET}"
echo "RUN_WINDOW_PROMPT=${RUN_WINDOW_PROMPT}"
echo "RUN_CONTINUITY=${RUN_CONTINUITY}"
echo "VIDEOBENCH_CONFIG_PATH=${VIDEOBENCH_CONFIG_PATH}"
echo "SEQUENCE_MODE=${SEQUENCE_MODE}"

exec bash "${combined_script}"
