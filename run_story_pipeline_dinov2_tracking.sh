#!/bin/bash -l
#SBATCH --job-name=sceneweaver_dino_track
#SBATCH --output=slurm_logs/sceneweaver_dino_track_%j.out
#SBATCH --error=slurm_logs/sceneweaver_dino_track_%j.err
#SBATCH --time=06:50:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16

set -euo pipefail

SCRIPT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# Route through the existing DINO-capable identity/layout pipeline without
# modifying the current stateful launcher.
export EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-dinov2}"
export LAST_FRAME_MEMORY="${LAST_FRAME_MEMORY:-1}"
export ENVIRONMENT_MEMORY="${ENVIRONMENT_MEMORY:-1}"
export CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES:-8}"
export CONTINUITY_MIN_SCORE="${CONTINUITY_MIN_SCORE:-0.76}"
export CONTINUITY_REGEN_ATTEMPTS="${CONTINUITY_REGEN_ATTEMPTS:-3}"
export TRANSITION_WEIGHT="${TRANSITION_WEIGHT:-0.70}"
export ENVIRONMENT_WEIGHT="${ENVIRONMENT_WEIGHT:-0.30}"
export IDENTITY_DRIFT_THRESHOLD="${IDENTITY_DRIFT_THRESHOLD:-0.74}"
export IDENTITY_PENALTY_SCALE="${IDENTITY_PENALTY_SCALE:-0.30}"
export LAYOUT_DRIFT_THRESHOLD="${LAYOUT_DRIFT_THRESHOLD:-0.68}"
export LAYOUT_PENALTY_SCALE="${LAYOUT_PENALTY_SCALE:-0.22}"
export CHARACTER_LOCK="${CHARACTER_LOCK:-keep the same named characters, face ids, wardrobe, and key props across windows; preserve anchor objects and their placement unless the beat clearly requires a change}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/story_run_dinov2_tracking_${RUN_STAMP}}"

exec "${SCRIPT_ROOT}/run_story_pipeline_identity_layout.sh"
