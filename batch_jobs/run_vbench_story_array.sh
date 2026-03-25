#!/bin/bash -l
#SBATCH --job-name=vbench_story_arr
#SBATCH --output=slurm_logs/vbench_story_arr_%A_%a.out
#SBATCH --error=slurm_logs/vbench_story_arr_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3%4

set -euo pipefail

PROJECT_ROOT="/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver"
STORY_VIDEOS_ROOT="${PROJECT_ROOT}/full_run/videos"
STORY_VIDEO_GLOB="*_full_story.mp4"

export PROJECT_ROOT
export STORY_VIDEOS_ROOT
export STORY_VIDEO_GLOB
export REQUIRE_WINDOW_PROMPT="${REQUIRE_WINDOW_PROMPT:-1}"

# Allow the combined launcher to install missing benchmark packages.
export INSTALL_VBENCH="${INSTALL_VBENCH:-1}"
export INSTALL_VIDEOBENCH="${INSTALL_VIDEOBENCH:-1}"
export UPGRADE_PIP="${UPGRADE_PIP:-0}"
export VBENCH_EXTRA_PIP_PACKAGES="${VBENCH_EXTRA_PIP_PACKAGES:-setuptools==59.6.0 scikit-build imageio imageio-ffmpeg}"
export VIDEOBENCH_EXTRA_PIP_PACKAGES="${VIDEOBENCH_EXTRA_PIP_PACKAGES:-}"

if [ "${RUN_WINDOW_PROMPT:-1}" = "1" ] && [ -z "${VIDEOBENCH_CONFIG_PATH:-}" ]; then
  echo "VIDEOBENCH_CONFIG_PATH is not set."
  echo "Set it when submitting the job if you want the Video-Bench window-prompt pass."
  echo "Example:"
  echo "  sbatch --export=ALL,VIDEOBENCH_CONFIG_PATH=/abs/path/to/config.json ${BASH_SOURCE[0]}"
  echo
  echo "If you only want VBench continuity, disable the Video-Bench half:"
  echo "  sbatch --export=ALL,RUN_WINDOW_PROMPT=0 ${BASH_SOURCE[0]}"
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs

exec bash "${PROJECT_ROOT}/run_vbench_combined_array.sh"
