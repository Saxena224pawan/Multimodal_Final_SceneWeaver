#!/bin/bash -l
#SBATCH --job-name=vbench_all_arr
#SBATCH --output=slurm_logs/vbench_all_arr_%A_%a.out
#SBATCH --error=slurm_logs/vbench_all_arr_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3%4

set -euo pipefail

COMMON_SLURM_ROOT="${COMMON_SLURM_ROOT:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}}"
COMMON_SLURM_SH="${COMMON_SLURM_ROOT}/slurm_common.sh"
# shellcheck source=./slurm_common.sh
source "${COMMON_SLURM_SH}"

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
COMBINED_SCRIPT="${COMBINED_SCRIPT:-${PROJECT_ROOT}/run_vbench_combined.sh}"
STORY_VIDEOS_ROOT="${STORY_VIDEOS_ROOT:-${PROJECT_ROOT}/full_run/videos}"
STORY_VIDEO_GLOB="${STORY_VIDEO_GLOB:-*_full_story.mp4}"
ARRAY_INDEX="${ARRAY_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
cd "${PROJECT_ROOT}"

if [ ! -x "${COMBINED_SCRIPT}" ]; then
  echo "Combined launcher not found or not executable: ${COMBINED_SCRIPT}"
  exit 1
fi

if [ ! -d "${STORY_VIDEOS_ROOT}" ]; then
  echo "STORY_VIDEOS_ROOT does not exist: ${STORY_VIDEOS_ROOT}"
  exit 1
fi

mapfile -t STORY_VIDEO_FILES < <(find "${STORY_VIDEOS_ROOT}" -maxdepth 1 -type f -name "${STORY_VIDEO_GLOB}" | sort)

if [ "${#STORY_VIDEO_FILES[@]}" -eq 0 ]; then
  echo "No story videos found in ${STORY_VIDEOS_ROOT} matching ${STORY_VIDEO_GLOB}"
  exit 1
fi

if [ "${ARRAY_INDEX}" -lt 0 ] || [ "${ARRAY_INDEX}" -ge "${#STORY_VIDEO_FILES[@]}" ]; then
  echo "Array index ${ARRAY_INDEX} is out of range for ${#STORY_VIDEO_FILES[@]} story videos."
  echo "Nothing to do."
  exit 0
fi

TARGET_VIDEO="${STORY_VIDEO_FILES[${ARRAY_INDEX}]}"

echo "STORY_VIDEOS_ROOT=${STORY_VIDEOS_ROOT}"
echo "STORY_VIDEO_GLOB=${STORY_VIDEO_GLOB}"
echo "ARRAY_INDEX=${ARRAY_INDEX}"
echo "STORY_COUNT=${#STORY_VIDEO_FILES[@]}"
echo "TARGET_VIDEO=${TARGET_VIDEO}"
echo "COMBINED_SCRIPT=${COMBINED_SCRIPT}"

export EVAL_TARGET="${TARGET_VIDEO}"
export REQUIRE_WINDOW_PROMPT="${REQUIRE_WINDOW_PROMPT:-0}"

exec bash "${COMBINED_SCRIPT}"
