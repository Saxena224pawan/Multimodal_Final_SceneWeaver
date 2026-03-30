#!/bin/bash -l
#SBATCH --job-name=sceneweaver_hindu_hq
#SBATCH --output=slurm_logs/sceneweaver_hindu_hq_%j.out
#SBATCH --error=slurm_logs/sceneweaver_hindu_hq_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export PIPELINE_MODE=agents
export STORYLINE="Two people respectfully discuss Hindu culture in the same temple courtyard."
export WINDOW_PLAN_JSON="$project_root/configs/window_plans/hindu_culture_dialogue.json"
export INITIAL_IMAGE="$project_root/default_start_image.png"
export RUN_NAME_PREFIX="hindu_dialogue_core_a100"
export TOTAL_MINUTES="0.8"
export WINDOW_SECONDS="8"
export FPS="8"
export NUM_FRAMES="65"
export STEPS="30"
export GUIDANCE_SCALE="5.2"
export CONTINUITY_CANDIDATES="2"
export CONTINUITY_MIN_SCORE="0.76"
export CONTINUITY_REGEN_ATTEMPTS="2"
export CRITIC_STORY_WEIGHT="0.72"
export TRANSITION_WEIGHT="0.58"
export ENVIRONMENT_WEIGHT="0.42"
export VISUAL_QUALITY_WEIGHT="0.18"
export STYLE_WEIGHT="0.24"
export REFERENCE_TAIL_FRAMES="4"
export REFERENCE_STRENGTH="0.72"
export STYLE_PREFIX="cinematic realistic dialogue, expressive two-person conversation, stable temple courtyard, rich stone texture, warm lamp light, coherent motion, consistent faces, detailed eyes and lips, restrained camera, high detail"
export NEGATIVE_PROMPT="blurry, low quality, flicker, frame jitter, deformed anatomy, duplicate subjects, extra people, inconsistent face identity, abrupt scene change, camera whip, text, subtitles, watermark, logo, collage, split-screen, glitch"
export CHARACTER_LOCK="keep the same two speakers across all windows, in the same temple courtyard, with stable clothing, face identity, spatial positions, and background landmarks."

bash "$project_root/run_story_pipeline.sh"
