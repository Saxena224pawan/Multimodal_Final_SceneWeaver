#!/bin/bash -l
#SBATCH --job-name=sceneweaver_lionmouse
#SBATCH --output=slurm_logs/sceneweaver_lionmouse_%j.out
#SBATCH --error=slurm_logs/sceneweaver_lionmouse_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$project_root"

export PIPELINE_MODE=core
export STORYLINE="A lion spares a mouse, later becomes trapped in a net, and the mouse returns to free the lion."
export WINDOW_PLAN_JSON="$project_root/configs/window_plans/lion_and_mouse_story.json"
export INITIAL_IMAGE="$project_root/default_start_image.png"
export RUN_NAME_PREFIX="lion_mouse_core_a100"
export TOTAL_MINUTES="0.8"
export WINDOW_SECONDS="8"
export FPS="8"
export NUM_FRAMES="65"
export STEPS="30"
export GUIDANCE_SCALE="5.3"
export CONTINUITY_CANDIDATES="2"
export CONTINUITY_MIN_SCORE="0.76"
export CONTINUITY_REGEN_ATTEMPTS="2"
export CRITIC_STORY_WEIGHT="0.72"
export TRANSITION_WEIGHT="0.58"
export ENVIRONMENT_WEIGHT="0.42"
export VISUAL_QUALITY_WEIGHT="0.18"
export STYLE_WEIGHT="0.24"
export REFERENCE_TAIL_FRAMES="4"
export REFERENCE_STRENGTH="0.70"
export STYLE_PREFIX="cinematic animal fable realism, expressive lion and mouse, warm forest light, stable woodland background, detailed fur, coherent motion, readable emotions, storybook quality, high detail"
export NEGATIVE_PROMPT="blurry, flicker, extra animals, duplicate lion, duplicate mouse, deformed anatomy, broken limbs, background change, watermark, text, logo, collage, humans"
export CHARACTER_LOCK="keep one golden lion and one tiny brown mouse consistent across all windows, with the same forest clearing, the same tree, and the same warm daylight."

bash "$project_root/run_story_pipeline.sh"
