#!/bin/bash -l
#SBATCH --job-name=vbench_pavan_sfs
#SBATCH --output=slurm_logs/vbench_pavan_sfs_%A_%a.out
#SBATCH --error=slurm_logs/vbench_pavan_sfs_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3%4

set -euo pipefail

project_root="${PROJECT_ROOT:-/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver}"
combined_script="${COMBINED_SCRIPT:-${project_root}/run_vbench_combined.sh}"
runs_root="${RUNS_ROOT:-${project_root}/outputs/story_runs_origin_pavan}"
report_root_name="${REPORT_ROOT_NAME:-benchmark_reports_fullstory}"
run_glob_override="${RUN_GLOB:-}"
videobench_config_path="${VIDEOBENCH_CONFIG_PATH:-${project_root}/configs/videobench_local_config.json}"
array_index="${SLURM_ARRAY_TASK_ID:-0}"
mode="${MODE:-simple}"

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

story_key="${story_keys[$array_index]}"
run_glob="${run_glob_override:-${mode}_${story_key}_origin_pavan_${mode}_*}"
target_run_dir=""
while IFS= read -r candidate; do
  [ -n "${candidate}" ] || continue
  if [ -f "${candidate}/${story_key}_full_story.mp4" ]; then
    target_run_dir="${candidate}"
    break
  fi
done < <(find "${runs_root}" -maxdepth 1 -type d -name "${run_glob}" | sort -r)
[ -n "${target_run_dir}" ] || { echo "No concat ${mode} run with ${story_key}_full_story.mp4 found in ${runs_root}"; exit 1; }

target_name="$(basename "${target_run_dir}")"
full_story_video="${target_run_dir}/${story_key}_full_story.mp4"
report_root_default="${target_run_dir}/${report_root_name}"

[ -x "${combined_script}" ] || { echo "Combined benchmark launcher not found or not executable: ${combined_script}"; exit 1; }
[ -d "${target_run_dir}" ] || { echo "Run directory does not exist: ${target_run_dir}"; exit 1; }
[ -f "${full_story_video}" ] || { echo "Missing concatenated full-story video: ${full_story_video}"; exit 1; }
[ -f "${videobench_config_path}" ] || { echo "Video-Bench config missing at ${videobench_config_path}"; exit 1; }

mkdir -p "${project_root}/slurm_logs"
mkdir -p "${report_root_default}"
cd "${project_root}"

export EVAL_TARGET="${target_run_dir}"
export STORY_RUN_DIR="${target_run_dir}"
export WINDOW_VIDEOS_PATH="${full_story_video}"
export CONTINUITY_VIDEOS_PATH="${full_story_video}"
export STORY_SLUG="${STORY_SLUG:-${target_name}}"
export RUN_NAME_BASE="${RUN_NAME_BASE:-${target_name}_fullstory}"
export REPORT_ROOT="${REPORT_ROOT:-${report_root_default}}"
export ENV_PATH="${ENV_PATH:-sceneweaver_runtime}"
export CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
export INSTALL_VBENCH="${INSTALL_VBENCH:-1}"
export INSTALL_VIDEOBENCH="${INSTALL_VIDEOBENCH:-1}"
export VBENCH_PACKAGE="${VBENCH_PACKAGE:-vbench}"
export VIDEOBENCH_PACKAGE="${VIDEOBENCH_PACKAGE:-git+https://github.com/Video-Bench/Video-Bench.git}"
export VBENCH_EXTRA_PIP_PACKAGES="${VBENCH_EXTRA_PIP_PACKAGES:-setuptools==59.6.0 scikit-build imageio imageio-ffmpeg}"
export RUN_WINDOW_PROMPT="${RUN_WINDOW_PROMPT:-1}"
export REQUIRE_WINDOW_PROMPT="${REQUIRE_WINDOW_PROMPT:-1}"
export RUN_CONTINUITY="${RUN_CONTINUITY:-1}"
export VIDEOBENCH_CONFIG_PATH="${videobench_config_path}"
export START_LOCAL_VIDEOBENCH_SERVER="${START_LOCAL_VIDEOBENCH_SERVER:-1}"
export LOCAL_SERVER_WAIT_SECONDS="${LOCAL_SERVER_WAIT_SECONDS:-3600}"
export LOCAL_SERVER_POLL_INTERVAL="${LOCAL_SERVER_POLL_INTERVAL:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SEQUENCE_MODE="${SEQUENCE_MODE:-concat_windows}"
export USE_PROXY="${USE_PROXY:-1}"
export PROXY_URL="${PROXY_URL:-http://proxy:80}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"

echo "PROJECT_ROOT=${project_root}"
echo "MODE=${mode}"
echo "STORY_KEY=${story_key}"
echo "TARGET_RUN_DIR=${target_run_dir}"
echo "WINDOW_VIDEOS_PATH=${WINDOW_VIDEOS_PATH}"
echo "CONTINUITY_VIDEOS_PATH=${CONTINUITY_VIDEOS_PATH}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "VIDEOBENCH_CONFIG_PATH=${VIDEOBENCH_CONFIG_PATH}"
echo "START_LOCAL_VIDEOBENCH_SERVER=${START_LOCAL_VIDEOBENCH_SERVER}"
echo "SEQUENCE_MODE=${SEQUENCE_MODE}"

exec bash "${combined_script}"
