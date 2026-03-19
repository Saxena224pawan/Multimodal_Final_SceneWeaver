#!/bin/bash -l
#SBATCH --job-name=sceneweaver_foxgrapes
#SBATCH --output=slurm_logs/sceneweaver_foxgrapes_%j.out
#SBATCH --error=slurm_logs/sceneweaver_foxgrapes_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$project_root"

export PIPELINE_MODE=core
export STORYLINE="A fox spots ripe grapes, keeps trying to reach them, fails, and finally walks away pretending the grapes are sour."
export WINDOW_PLAN_JSON="$project_root/configs/window_plans/fox_and_grapes_story.json"
export INITIAL_IMAGE="$project_root/default_start_image.png"
export RUN_NAME_PREFIX="fox_grapes_core_a100"
export TOTAL_MINUTES="0.8"
export WINDOW_SECONDS="8"
export FPS="8"
export NUM_FRAMES="65"
export STEPS="30"
export GUIDANCE_SCALE="5.3"
export CONTINUITY_CANDIDATES="2"
export CONTINUITY_MIN_SCORE="0.76"
export CONTINUITY_REGEN_ATTEMPTS="2"
export CRITIC_STORY_WEIGHT="0.71"
export TRANSITION_WEIGHT="0.58"
export ENVIRONMENT_WEIGHT="0.42"
export VISUAL_QUALITY_WEIGHT="0.18"
export STYLE_WEIGHT="0.24"
export REFERENCE_TAIL_FRAMES="4"
export REFERENCE_STRENGTH="0.70"
export STYLE_PREFIX="cinematic storybook realism, single red fox, lush vineyard, ripe grapes, warm golden-hour light, stable camera, coherent motion, detailed fur, high detail"
export NEGATIVE_PROMPT="blurry, flicker, duplicate fox, extra animals, deformed anatomy, missing grapes, background change, watermark, text, logo, collage, humans"
export CHARACTER_LOCK="keep one single red fox consistent across all windows, with the same vineyard rows, the same grape cluster, and the same golden afternoon light."

bash "$project_root/run_story_pipeline.sh"
