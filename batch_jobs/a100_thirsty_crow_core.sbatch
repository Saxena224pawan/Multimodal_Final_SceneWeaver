#!/bin/bash -l
#SBATCH --job-name=sceneweaver_crow_hq
#SBATCH --output=slurm_logs/sceneweaver_crow_hq_%j.out
#SBATCH --error=slurm_logs/sceneweaver_crow_hq_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$project_root"

export PIPELINE_MODE=core
export STORYLINE="A thirsty crow searches for water, finds a clay pot, raises the water by dropping stones into it, and finally drinks."
export WINDOW_PLAN_JSON="$project_root/configs/window_plans/thirsty_crow_story.json"
export INITIAL_IMAGE="$project_root/thirsty_crow_start_image.png"
export RUN_NAME_PREFIX="thirsty_crow_core_a100"
export TOTAL_MINUTES="0.8"
export WINDOW_SECONDS="8"
export FPS="8"
export NUM_FRAMES="65"
export STEPS="30"
export GUIDANCE_SCALE="5.4"
export CONTINUITY_CANDIDATES="2"
export CONTINUITY_MIN_SCORE="0.76"
export CONTINUITY_REGEN_ATTEMPTS="2"
export CRITIC_STORY_WEIGHT="0.70"
export TRANSITION_WEIGHT="0.60"
export ENVIRONMENT_WEIGHT="0.40"
export VISUAL_QUALITY_WEIGHT="0.18"
export STYLE_WEIGHT="0.25"
export REFERENCE_TAIL_FRAMES="4"
export REFERENCE_STRENGTH="0.68"
export STYLE_PREFIX="cinematic storybook realism, single black crow, expressive eyes, detailed feathers, clear beak actions, stable camera, coherent motion, detailed clay pot, visible stones, warm daylight, high detail"
export NEGATIVE_PROMPT="blurry, flicker, duplicate bird, extra birds, multiple crows, deformed beak, broken pot, background change, watermark, text, logo, collage, humans"
export CHARACTER_LOCK="keep one black crow consistent across all windows, with the same clay pot and the same dusty courtyard under the same tree."

bash "$project_root/run_story_pipeline.sh"
