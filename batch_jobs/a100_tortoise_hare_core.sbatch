#!/bin/bash -l
#SBATCH --job-name=sceneweaver_torthare
#SBATCH --output=slurm_logs/sceneweaver_torthare_%j.out
#SBATCH --error=slurm_logs/sceneweaver_torthare_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$project_root"

export PIPELINE_MODE=agents
export STORYLINE="A hare races ahead, falls asleep, and loses to the steady tortoise on the same meadow path."
export WINDOW_PLAN_JSON="$project_root/configs/window_plans/tortoise_and_hare_story.json"
export INITIAL_IMAGE="$project_root/default_start_image.png"
export RUN_NAME_PREFIX="tortoise_hare_agents_a100"
export TOTAL_MINUTES="0.8"
export WINDOW_SECONDS="8"
export FPS="8"
export NUM_FRAMES="65"
export STEPS="30"
export GUIDANCE_SCALE="5.3"
export CONTINUITY_CANDIDATES="2"
export CONTINUITY_MIN_SCORE="0.76"
export CONTINUITY_REGEN_ATTEMPTS="4"
export CRITIC_STORY_WEIGHT="0.80"
export TRANSITION_WEIGHT="0.80"
export ENVIRONMENT_WEIGHT="0.50"
export VISUAL_QUALITY_WEIGHT="0.50"
export STYLE_WEIGHT="0.24"
export AGENT_MAX_ITERATIONS="10"
export AGENT_QUALITY_THRESHOLD="0.90"
export REFERENCE_TAIL_FRAMES="4"
export REFERENCE_STRENGTH="0.70"
export STYLE_PREFIX="cinematic meadow fable realism, expressive hare and tortoise, bright daylight, stable race path, coherent motion, detailed fur and shell, readable emotions, high detail"
export NEGATIVE_PROMPT="blurry, flicker, duplicate hare, duplicate tortoise, extra animals, deformed anatomy, background change, watermark, text, logo, collage, humans"
export CHARACTER_LOCK="keep one gray hare and one green-brown tortoise consistent across all windows, with the same meadow path, same distant trees, and same bright daylight."

bash "$project_root/run_story_pipeline.sh"
