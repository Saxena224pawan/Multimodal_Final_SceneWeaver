#!/bin/bash -l
#SBATCH --job-name=vbench_fox_grapes
#SBATCH --output=../slurm_logs/vbench_fox_grapes_%j.out
#SBATCH --error=../slurm_logs/vbench_fox_grapes_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_ROOT="/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver"
STORY_KEY="fox_and_grapes"
STORY_RUN_DIR="/home/vault/v123be/v123be36/sceneweaver_runs/fox_and_grapes_260319_225547"
STORY_CONFIG_SH="${PROJECT_ROOT}/configs/stories/short_stories.sh"
TARGET_VIDEO="${PROJECT_ROOT}/full_run/videos/${STORY_KEY}_full_story.mp4"

# shellcheck source=../configs/stories/short_stories.sh
source "${STORY_CONFIG_SH}"
[ -n "${SHORT_STORIES[${STORY_KEY}]+x}" ] || { echo "Unknown story key: ${STORY_KEY}"; exit 1; }
[ -f "${TARGET_VIDEO}" ] || { echo "Missing story video: ${TARGET_VIDEO}"; exit 1; }
[ -d "${STORY_RUN_DIR}/clips" ] || { echo "Missing clips dir in STORY_RUN_DIR: ${STORY_RUN_DIR}"; exit 1; }
[ -f "${STORY_RUN_DIR}/run_log.jsonl" ] || { echo "Missing run_log.jsonl in STORY_RUN_DIR: ${STORY_RUN_DIR}"; exit 1; }

export PROJECT_ROOT
export EVAL_TARGET="${TARGET_VIDEO}"
export STORY_SLUG="${STORY_KEY}"
export STORY_RUN_DIR
export REQUIRE_WINDOW_PROMPT="${REQUIRE_WINDOW_PROMPT:-1}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs

exec bash "${PROJECT_ROOT}/run_vbench_combined.sh"
