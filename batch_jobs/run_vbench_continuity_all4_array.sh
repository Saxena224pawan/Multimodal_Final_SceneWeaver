#!/bin/bash -l
#SBATCH --job-name=vbench_cont_4
#SBATCH --output=slurm_logs/vbench_cont_4_%A_%a.out
#SBATCH --error=slurm_logs/vbench_cont_4_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3%4

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver}"
STORY_VIDEOS_ROOT="${STORY_VIDEOS_ROOT:-${PROJECT_ROOT}/full_run/videos}"
REPORT_ROOT_BASE="${REPORT_ROOT_BASE:-${PROJECT_ROOT}/outputs/reports/vbench_continuity_all4}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-vbench_continuity}"

story_keys=(
  "fox_and_grapes"
  "lion_and_mouse"
  "thirsty_crow"
  "tortoise_and_hare"
)

task_index="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_index}" -lt 0 ] || [ "${task_index}" -ge "${#story_keys[@]}" ]; then
  echo "Array index ${task_index} is out of range for ${#story_keys[@]} stories."
  exit 1
fi

story_key="${story_keys[$task_index]}"
video_path="${STORY_VIDEOS_ROOT}/${story_key}_full_story.mp4"

[ -f "${video_path}" ] || { echo "Missing story video: ${video_path}"; exit 1; }

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${REPORT_ROOT_BASE}"
cd "${PROJECT_ROOT}"

export PROJECT_ROOT
export VIDEOS_PATH="${video_path}"
export REPORT_ROOT="${REPORT_ROOT_BASE}"
export RUN_NAME="${RUN_NAME_PREFIX}_${story_key}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "STORY_KEY=${story_key}"
echo "VIDEOS_PATH=${VIDEOS_PATH}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"

exec bash "${PROJECT_ROOT}/run_vbench_continuity.sh"
