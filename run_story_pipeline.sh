#!/bin/bash -l
#SBATCH --job-name=sceneweaver_crow
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
<<<<<<< Updated upstream
#SBATCH --time=6:50:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
=======
#SBATCH --time=04:30:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
>>>>>>> Stashed changes

set -euo pipefail

################################
# project setup
################################

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_root}"

<<<<<<< Updated upstream
ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-sceneweaver_runtime}"

WAN_LOCAL_MODEL="${WAN_LOCAL_MODEL:-/home/vault/v123be/v123be36/Wan2.1-T2V-1.3B-Diffusers}"
MODEL_STORAGE_ROOT="${MODEL_STORAGE_ROOT:-/home/vault/v123be/v123be36}"
=======
mkdir -p slurm_logs
mkdir -p outputs
mkdir -p .hf

hpc_vault_root="/home/vault/v123be/v123be37"
sceneweaver_runs_root="${hpc_vault_root}/sceneweaver_runs"
mkdir -p "${sceneweaver_runs_root}"
>>>>>>> Stashed changes

echo "===================================="
echo "SceneWeaver job started"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Start time: $(date)"
echo "Project root: $project_root"
echo "===================================="

################################
# GPU visibility
################################

<<<<<<< Updated upstream
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_REPO="${MODEL_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/$(basename "${MODEL_REPO}")}"
MODEL_INCLUDE="${MODEL_INCLUDE:-*.safetensors *.json *.txt tokenizer* *.model}"
DOWNLOAD_DIRECTOR_MODEL="${DOWNLOAD_DIRECTOR_MODEL:-0}"
DIRECTOR_MODEL_REPO="${DIRECTOR_MODEL_REPO:-Qwen/Qwen2.5-3B-Instruct}"
DIRECTOR_MODEL_DIR="${DIRECTOR_MODEL_DIR:-${MODEL_STORAGE_ROOT}/$(basename "${DIRECTOR_MODEL_REPO}")}"
DIRECTOR_MODEL_INCLUDE="${DIRECTOR_MODEL_INCLUDE:-*.safetensors *.json *.txt tokenizer* *.model}"

DINOV2_LOCAL_MODEL="${DINOV2_LOCAL_MODEL:-${MODEL_STORAGE_ROOT}/dinov2-base}"
=======
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

################################
# model locations
################################
>>>>>>> Stashed changes

wan_model="${hpc_vault_root}/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers"
director_model="${hpc_vault_root}/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct"
dinov2_model="${hpc_vault_root}/facebook/dinov2-base"
window_plan_json="${project_root}/configs/window_plans/thirsty_crow_story.json"

################################
# safety checks
################################

<<<<<<< Updated upstream
# Default to lit_eval when neither ENV_PATH nor VENV_PATH is provided.
if [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ]; then
  ENV_PATH="${DEFAULT_ENV_PATH}"
fi
=======
echo "[CHECK] verifying model paths..."

[ -d "$wan_model" ] || { echo "ERROR: Wan model missing at $wan_model"; exit 1; }
[ -d "$director_model" ] || { echo "ERROR: Director model missing at $director_model"; exit 1; }
[ -d "$dinov2_model" ] || { echo "ERROR: DINOv2 model missing at $dinov2_model"; exit 1; }
[ -f "$window_plan_json" ] || { echo "ERROR: Window plan missing at $window_plan_json"; exit 1; }
>>>>>>> Stashed changes

echo "[CHECK] models found ✓"

################################
# python environment
################################

echo "[INFO] activating conda environment..."

source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate sceneweaver311

python_bin=$(which python)
echo "[INFO] python executable: $python_bin"

export PYTHONUNBUFFERED=1
export HF_HOME="${project_root}/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

################################
# generation parameters
################################

total_minutes=0.8
window_seconds=8

fps=8
num_frames=65
steps=28
guidance_scale=5.4

height=480
width=832

seed=42
device=cuda

################################
# continuity parameters
################################

continuity_candidates=2
continuity_min_score=0.74
continuity_regen_attempts=2

critic_story_weight=0.70
transition_weight=0.65
environment_weight=0.35
visual_quality_weight=0.18

################################
# output directory
################################

run_stamp=$(date +%y%m%d_%H%M%S)
output_dir="${sceneweaver_runs_root}/thirsty_crow_${run_stamp}"

mkdir -p "$output_dir"

echo "[INFO] output directory:"
echo "$output_dir"

################################
# storyline
################################

storyline="A thirsty crow searches for water, finds a clay pot, raises the water by dropping stones into it, and finally drinks."

################################
# style
################################

style_prefix="cinematic storybook realism, single black crow, expressive eyes, detailed feathers, clear beak actions, stable camera, coherent motion, detailed clay pot, visible stones, warm daylight, high detail"
negative_prompt="blurry, flicker, duplicate bird, extra birds, multiple crows, deformed beak, broken pot, background change, watermark, text, logo, collage, humans"
character_lock="keep one black crow consistent across all windows, with the same clay pot and the same dusty courtyard under the same tree."

################################
# reference conditioning
################################

reference_strength=0.68
reference_tail_frames=4

################################
# initial reference frame
################################

initial_image="${project_root}/thirsty_crow_start_image.png"

if [ ! -f "$initial_image" ]; then
echo "[INFO] generating thirsty crow start image..."

"$python_bin" <<'PYEOF'
from PIL import Image, ImageDraw

img = Image.new("RGB", (832, 480), (214, 188, 140))
d = ImageDraw.Draw(img)

d.ellipse((60, 20, 790, 430), fill=(224, 205, 160))
d.ellipse((520, 80, 740, 300), fill=(110, 90, 60))
d.rectangle((0, 360, 832, 480), fill=(191, 155, 102))

d.ellipse((500, 30, 780, 210), fill=(110, 120, 75))
d.rectangle((620, 120, 650, 360), fill=(108, 85, 50))

d.ellipse((250, 140, 470, 390), fill=(166, 100, 52))
d.rectangle((295, 200, 425, 390), fill=(166, 100, 52))
d.ellipse((280, 120, 440, 230), fill=(189, 121, 68))
d.ellipse((308, 215, 412, 320), fill=(70, 110, 145))

d.ellipse((120, 180, 240, 250), fill=(25, 25, 25))
d.polygon([(230, 210), (280, 195), (238, 228)], fill=(210, 160, 70))
d.polygon([(105, 205), (78, 175), (95, 228)], fill=(18, 18, 18))
d.polygon([(165, 246), (150, 300), (188, 248)], fill=(18, 18, 18))
d.polygon([(192, 246), (182, 303), (220, 250)], fill=(18, 18, 18))
d.ellipse((205, 198, 215, 208), fill=(240, 240, 220))

for x, y in ((510, 350), (545, 368), (585, 344), (622, 364), (662, 342), (700, 360)):
    d.ellipse((x, y, x + 18, y + 14), fill=(120, 112, 102))

img.save("thirsty_crow_start_image.png")
PYEOF
fi

################################
# build argument list
################################

<<<<<<< Updated upstream
# STORYLINE
if [ -z "${STORYLINE:-}" ]; then
  STORYLINE="$(cat <<'STORY_EOF'
A thirsty crow finds a pot with very little water. The crow carries small stones one by one and drops them into the pot. As more stones fall in, the water level rises gradually. At last, the crow can reach the water and drinks.
STORY_EOF
)"
fi

if [ -z "${STYLE_PREFIX:-}" ]; then
  STYLE_PREFIX="$(cat <<'STYLE_EOF'
cinematic realistic drama, natural skin tones, stable camera, coherent motion, soft depth of field, high detail, grounded everyday realism
STYLE_EOF
)"
fi

if [ -z "${CHARACTER_LOCK:-}" ]; then
  CHARACTER_LOCK="$(cat <<'CHAR_EOF'
exactly one crow with consistent appearance across all windows: glossy black feathers, medium size, sharp beak, bright eyes. Keep one clay pot and small pebbles/stones consistent in shape and color. No extra animals, no humans, no crowds, no duplicate or cloned crows.
CHAR_EOF
)"
fi

if [ -z "${GLOBAL_CONTINUITY_ANCHOR:-}" ]; then
  GLOBAL_CONTINUITY_ANCHOR="$(cat <<'ANCHOR_EOF'
All windows occur in the same outdoor courtyard. Anchor objects must stay: one clay pot on a small wooden table, scattered pebbles near the table, one tree branch where the crow perches, warm daylight. Keep identical courtyard layout, object positions, and lighting progression. No random location changes, no teleporting objects, no sky-falling stones.
ANCHOR_EOF
)"
fi

if [ -z "${NEGATIVE_PROMPT:-}" ]; then
  NEGATIVE_PROMPT="$(cat <<'NEG_EOF'
blurry, low quality, flicker, frame jitter, deformed anatomy, extra limbs, duplicate people, cloned faces, twin characters, repeated character in frame, crowd, background people, strangers, extra humans, extra children, extra women, extra men, inconsistent face, identity drift, face morph, different person, different outfit, wardrobe change, different hairstyle, different location, different room layout, different furniture, inconsistent background, scene change, teleport, background swap, text overlay, subtitles, watermark, logo, collage, split-screen, glitch
NEG_EOF
)"
fi

# Shot planning defaults
SHOT_PLAN_ENFORCE="${SHOT_PLAN_ENFORCE:-1}"
SHOT_PLAN_DEFAULTS="${SHOT_PLAN_DEFAULTS:-cinematic}"
WINDOW_PLAN_JSON="${WINDOW_PLAN_JSON:-}"

CAPTIONER_MODEL_ID="${CAPTIONER_MODEL_ID:-}"
CAPTIONER_DEVICE="${CAPTIONER_DEVICE:-cpu}"
CAPTIONER_STUB_FALLBACK="${CAPTIONER_STUB_FALLBACK:-1}"

# Track which knobs were explicitly provided by user/environment so presets
# only override unset values.
HAS_FPS="${FPS+x}"
HAS_NUM_FRAMES="${NUM_FRAMES+x}"
HAS_STEPS="${STEPS+x}"
HAS_CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES+x}"
HAS_CONTINUITY_REGEN_ATTEMPTS="${CONTINUITY_REGEN_ATTEMPTS+x}"
HAS_CONTINUITY_MIN_SCORE="${CONTINUITY_MIN_SCORE+x}"
HAS_RUN_REPAIR_PASS="${RUN_REPAIR_PASS+x}"
HAS_REPAIR_CANDIDATES="${REPAIR_CANDIDATES+x}"
HAS_REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS+x}"
HAS_PARALLEL_WINDOW_MODE="${PARALLEL_WINDOW_MODE+x}"

TOTAL_MINUTES="${TOTAL_MINUTES:-1}"
FPS="${FPS:-12}"
WINDOW_SECONDS="${WINDOW_SECONDS:-8}"
NUM_FRAMES="${NUM_FRAMES:-96}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SEED="${SEED:-42}"
SEED_STRATEGY="${SEED_STRATEGY:-fixed}"

if [ -z "${DIRECTOR_MODEL_ID:-}" ] && [ -d "${DIRECTOR_MODEL_DIR}" ]; then
  DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_DIR}"
else
  DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_ID:-}"
fi
DIRECTOR_TEMPERATURE="${DIRECTOR_TEMPERATURE:-0.05}"
if [ -z "${DIRECTOR_MAX_NEW_TOKENS:-}" ]; then
  if echo "${DIRECTOR_MODEL_ID}" | grep -qi "qwen"; then
    DIRECTOR_MAX_NEW_TOKENS="128"
  else
    DIRECTOR_MAX_NEW_TOKENS="160"
  fi
fi
DIRECTOR_DO_SAMPLE="${DIRECTOR_DO_SAMPLE:-0}"

EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-dinov2}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-${DINOV2_LOCAL_MODEL}}"
LAST_FRAME_MEMORY="${LAST_FRAME_MEMORY:-1}"
CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES:-8}"
CONTINUITY_MIN_SCORE="${CONTINUITY_MIN_SCORE:-0.72}"
CONTINUITY_REGEN_ATTEMPTS="${CONTINUITY_REGEN_ATTEMPTS:-2}"
CRITIC_STORY_WEIGHT="${CRITIC_STORY_WEIGHT:-0.15}"
ENVIRONMENT_MEMORY="${ENVIRONMENT_MEMORY:-1}"
TRANSITION_WEIGHT="${TRANSITION_WEIGHT:-0.65}"
ENVIRONMENT_WEIGHT="${ENVIRONMENT_WEIGHT:-0.35}"
SCENE_CHANGE_ENV_DECAY="${SCENE_CHANGE_ENV_DECAY:-0.25}"

EMBEDDING_ADAPTER_CKPT="${EMBEDDING_ADAPTER_CKPT:-${PROJECT_ROOT}/outputs/pororo_continuity_adapter.pt}"

DRY_RUN="${DRY_RUN:-0}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/story_run_${RUN_STAMP}}"
COMBINE_WINDOWS="${COMBINE_WINDOWS:-0}"
COMBINED_VIDEO_NAME="${COMBINED_VIDEO_NAME:-story_windows_concat.mp4}"
RUN_REPAIR_PASS="${RUN_REPAIR_PASS:-0}"
REPAIR_REPLACE_ORIGINAL="${REPAIR_REPLACE_ORIGINAL:-1}"
REPAIR_CANDIDATES="${REPAIR_CANDIDATES:-8}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-3}"
REPAIR_ACCEPT_SCORE="${REPAIR_ACCEPT_SCORE:-0.86}"
REPAIR_TRANSITION_THRESHOLD="${REPAIR_TRANSITION_THRESHOLD:-0.84}"
REPAIR_CRITIC_THRESHOLD="${REPAIR_CRITIC_THRESHOLD:-0.82}"
RUN_PRESET="${RUN_PRESET:-quality}"   # quality|fast
WINDOW_SHARD_COUNT="${WINDOW_SHARD_COUNT:-1}"
WINDOW_SHARD_INDEX="${WINDOW_SHARD_INDEX:-0}"
PARALLEL_WINDOW_MODE="${PARALLEL_WINDOW_MODE:-0}"
PARALLEL_GPUS_PER_NODE="${PARALLEL_GPUS_PER_NODE:-0}"

if [ "${RUN_PRESET}" = "fast" ]; then
  [ -z "${HAS_FPS}" ] && FPS="8"
  [ -z "${HAS_NUM_FRAMES}" ] && NUM_FRAMES="64"
  [ -z "${HAS_STEPS}" ] && STEPS="16"
  [ -z "${HAS_CONTINUITY_CANDIDATES}" ] && CONTINUITY_CANDIDATES="2"
  [ -z "${HAS_CONTINUITY_REGEN_ATTEMPTS}" ] && CONTINUITY_REGEN_ATTEMPTS="1"
  [ -z "${HAS_CONTINUITY_MIN_SCORE}" ] && CONTINUITY_MIN_SCORE="0.72"
  [ -z "${HAS_RUN_REPAIR_PASS}" ] && RUN_REPAIR_PASS="0"
  [ -z "${HAS_REPAIR_CANDIDATES}" ] && REPAIR_CANDIDATES="2"
  [ -z "${HAS_REPAIR_ATTEMPTS}" ] && REPAIR_ATTEMPTS="1"
  [ -z "${HAS_PARALLEL_WINDOW_MODE}" ] && PARALLEL_WINDOW_MODE="1"
fi

# Auto-bind shard index for SLURM array jobs when not explicitly set.
if [ "${WINDOW_SHARD_COUNT}" -gt 1 ] && [ "${WINDOW_SHARD_INDEX}" = "0" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  WINDOW_SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
fi

# Select model id
if [ -n "${WAN_LOCAL_MODEL}" ]; then
  DEFAULT_VIDEO_MODEL_ID="${WAN_LOCAL_MODEL}"
elif [ -d "${MODEL_DIR}" ]; then
  DEFAULT_VIDEO_MODEL_ID="${MODEL_DIR}"
else
  DEFAULT_VIDEO_MODEL_ID="${MODEL_REPO}"
fi
VIDEO_MODEL_ID="${VIDEO_MODEL_ID:-${DEFAULT_VIDEO_MODEL_ID}}"

if [ "${USE_OFFLINE_MODE}" = "1" ] && [ "${EMBEDDING_BACKEND}" = "dinov2" ]; then
  if [ ! -f "${EMBEDDING_MODEL_ID}/preprocessor_config.json" ]; then
    echo "Offline dinov2 embedding requires a local path with preprocessor_config.json."
    echo "Current EMBEDDING_MODEL_ID='${EMBEDDING_MODEL_ID}' is not a valid local dinov2 directory."
    exit 1
  fi
fi

if [ -d "${VIDEO_MODEL_ID}" ] && [ ! -f "${VIDEO_MODEL_ID}/model_index.json" ]; then
  echo "Local VIDEO_MODEL_ID is not a diffusers pipeline directory: ${VIDEO_MODEL_ID}"
  echo "Missing required file: ${VIDEO_MODEL_ID}/model_index.json"
  exit 1
fi

if echo "${VIDEO_MODEL_ID}" | grep -qi "TI2V"; then
  echo "VIDEO_MODEL_ID appears to be a TI2V model: ${VIDEO_MODEL_ID}"
  echo "Use a T2V model for text-only generation."
  exit 1
fi

# fail fast if adapter path is set but missing
if [ -n "${EMBEDDING_ADAPTER_CKPT}" ] && [ ! -f "${EMBEDDING_ADAPTER_CKPT}" ]; then
  echo "ERROR: embedding adapter checkpoint not found: ${EMBEDDING_ADAPTER_CKPT}"
  exit 1
fi

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python/python3 found."
    exit 1
  fi
fi

# Optional selective model download.
if [ "${DOWNLOAD_MODEL}" = "1" ] || [ "${DOWNLOAD_DIRECTOR_MODEL}" = "1" ]; then
  HF_DL=""
  if command -v hf >/dev/null 2>&1; then
    HF_DL="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DL="huggingface-cli"
  else
    echo "No HF CLI found. Install huggingface_hub[cli] or set model paths to local directories."
    exit 1
  fi
fi

if [ "${DOWNLOAD_MODEL}" = "1" ]; then
  mkdir -p "${MODEL_DIR}"
  # shellcheck disable=SC2206
  INCLUDE_ARR=(${MODEL_INCLUDE})
  HF_INCLUDE_ARGS=()
  for pat in "${INCLUDE_ARR[@]}"; do
    HF_INCLUDE_ARGS+=(--include "${pat}")
  done
  "${HF_DL}" download "${MODEL_REPO}" --local-dir "${MODEL_DIR}" "${HF_INCLUDE_ARGS[@]}"
  VIDEO_MODEL_ID="${MODEL_DIR}"
fi

if [ "${DOWNLOAD_DIRECTOR_MODEL}" = "1" ]; then
  mkdir -p "${DIRECTOR_MODEL_DIR}"
  # shellcheck disable=SC2206
  D_INCLUDE_ARR=(${DIRECTOR_MODEL_INCLUDE})
  D_HF_INCLUDE_ARGS=()
  for pat in "${D_INCLUDE_ARR[@]}"; do
    D_HF_INCLUDE_ARGS+=(--include "${pat}")
  done
  "${HF_DL}" download "${DIRECTOR_MODEL_REPO}" --local-dir "${DIRECTOR_MODEL_DIR}" "${D_HF_INCLUDE_ARGS[@]}"
  DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_DIR}"
fi

if [ "${DRY_RUN}" = "0" ]; then
  if [ "${DEVICE}" = "cuda" ] && ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "DEVICE=cuda requested, but torch.cuda.is_available() is false in this environment."
    if [ "${STRICT_DEVICE}" = "1" ]; then
      echo "STRICT_DEVICE=1 set; exiting."
      exit 1
    fi
    echo "Falling back to DEVICE=auto."
    DEVICE="auto"
  fi
fi

SCRIPT_HELP="$("${PYTHON_BIN}" scripts/run_story_pipeline.py --help 2>/dev/null || true)"
supports_flag() {
  echo "${SCRIPT_HELP}" | grep -q -- "$1"
}

# Inject global continuity anchor into style context when supported.
STYLE_PREFIX_COMBINED="${STYLE_PREFIX} Environment anchor: ${GLOBAL_CONTINUITY_ANCHOR}"

CMD=("${PYTHON_BIN}" scripts/run_story_pipeline.py
  --storyline "${STORYLINE}"
  --total_minutes "${TOTAL_MINUTES}"
  --window_seconds "${WINDOW_SECONDS}"
  --video_model_id "${VIDEO_MODEL_ID}"
  --director_temperature "${DIRECTOR_TEMPERATURE}"
  --director_max_new_tokens "${DIRECTOR_MAX_NEW_TOKENS}"
  --embedding_backend "${EMBEDDING_BACKEND}"
  --embedding_model_id "${EMBEDDING_MODEL_ID}"
  --num_frames "${NUM_FRAMES}"
  --steps "${STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --fps "${FPS}"
  --seed "${SEED}"
  --seed_strategy "${SEED_STRATEGY}"
  --device "${DEVICE}"
  --output_dir "${OUTPUT_DIR}"
)

if supports_flag "--style_prefix"; then
  CMD+=(--style_prefix "${STYLE_PREFIX_COMBINED}")
fi
if supports_flag "--character_lock"; then
  CMD+=(--character_lock "${CHARACTER_LOCK}")
fi
if supports_flag "--negative_prompt"; then
  CMD+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi
if supports_flag "--embedding_adapter_ckpt" && [ -n "${EMBEDDING_ADAPTER_CKPT}" ]; then
  CMD+=(--embedding_adapter_ckpt "${EMBEDDING_ADAPTER_CKPT}")
fi
if supports_flag "--last_frame_memory" && [ "${LAST_FRAME_MEMORY}" = "1" ]; then
  CMD+=(--last_frame_memory)
fi
if supports_flag "--continuity_candidates"; then
  CMD+=(--continuity_candidates "${CONTINUITY_CANDIDATES}")
fi
if supports_flag "--continuity_min_score"; then
  CMD+=(--continuity_min_score "${CONTINUITY_MIN_SCORE}")
fi
if supports_flag "--continuity_regen_attempts"; then
  CMD+=(--continuity_regen_attempts "${CONTINUITY_REGEN_ATTEMPTS}")
fi
if supports_flag "--critic_story_weight"; then
  CMD+=(--critic_story_weight "${CRITIC_STORY_WEIGHT}")
fi
if supports_flag "--environment_memory"; then
  if [ "${ENVIRONMENT_MEMORY}" = "1" ]; then
    CMD+=(--environment_memory)
  else
    CMD+=(--no-environment_memory)
  fi
fi
if supports_flag "--transition_weight"; then
  CMD+=(--transition_weight "${TRANSITION_WEIGHT}")
fi
if supports_flag "--transition_floor" && [ -n "${TRANSITION_FLOOR:-}" ]; then
  CMD+=(--transition_floor "${TRANSITION_FLOOR}")
fi
if supports_flag "--environment_weight"; then
  CMD+=(--environment_weight "${ENVIRONMENT_WEIGHT}")
fi
if supports_flag "--scene_change_env_decay"; then
  CMD+=(--scene_change_env_decay "${SCENE_CHANGE_ENV_DECAY}")
fi
if supports_flag "--window_shard_count"; then
  CMD+=(--window_shard_count "${WINDOW_SHARD_COUNT}")
fi
if supports_flag "--window_shard_index"; then
  CMD+=(--window_shard_index "${WINDOW_SHARD_INDEX}")
fi
if supports_flag "--shot_plan_defaults"; then
  CMD+=(--shot_plan_defaults "${SHOT_PLAN_DEFAULTS}")
fi
if supports_flag "--shot_plan_enforce"; then
  if [ "${SHOT_PLAN_ENFORCE}" = "1" ]; then
    CMD+=(--shot_plan_enforce)
  else
    CMD+=(--no-shot_plan_enforce)
  fi
fi
if supports_flag "--window_plan_json" && [ -n "${WINDOW_PLAN_JSON}" ]; then
  CMD+=(--window_plan_json "${WINDOW_PLAN_JSON}")
fi
if supports_flag "--parallel_window_mode"; then
  if [ "${PARALLEL_WINDOW_MODE}" = "1" ]; then
    CMD+=(--parallel_window_mode)
  else
    CMD+=(--no-parallel_window_mode)
  fi
fi
if supports_flag "--captioner_model_id" && [ -n "${CAPTIONER_MODEL_ID}" ]; then
  CMD+=(--captioner_model_id "${CAPTIONER_MODEL_ID}")
  CMD+=(--captioner_device "${CAPTIONER_DEVICE}")
  if [ "${CAPTIONER_STUB_FALLBACK}" = "0" ]; then
    CMD+=(--no-captioner_stub_fallback)
  fi
fi
if [ -n "${DIRECTOR_MODEL_ID}" ]; then
  CMD+=(--director_model_id "${DIRECTOR_MODEL_ID}")
fi
if supports_flag "--director_do_sample"; then
  if [ "${DIRECTOR_DO_SAMPLE}" = "1" ]; then
    CMD+=(--director_do_sample)
  else
    CMD+=(--no-director_do_sample)
  fi
fi
if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry_run)
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEVICE=${DEVICE}"
echo "VIDEO_MODEL_ID=${VIDEO_MODEL_ID}"
echo "FPS=${FPS}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "WINDOW_SECONDS=${WINDOW_SECONDS}"
echo "SEED=${SEED}"
echo "SEED_STRATEGY=${SEED_STRATEGY}"
echo "DIRECTOR_MODEL_ID=${DIRECTOR_MODEL_ID}"
echo "DIRECTOR_TEMPERATURE=${DIRECTOR_TEMPERATURE}"
echo "DIRECTOR_MAX_NEW_TOKENS=${DIRECTOR_MAX_NEW_TOKENS}"
echo "DIRECTOR_DO_SAMPLE=${DIRECTOR_DO_SAMPLE}"
echo "EMBEDDING_BACKEND=${EMBEDDING_BACKEND}"
echo "SHOT_PLAN_ENFORCE=${SHOT_PLAN_ENFORCE}"
echo "SHOT_PLAN_DEFAULTS=${SHOT_PLAN_DEFAULTS}"
echo "WINDOW_PLAN_JSON=${WINDOW_PLAN_JSON}"
echo "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}"
echo "LAST_FRAME_MEMORY=${LAST_FRAME_MEMORY}"
echo "CONTINUITY_CANDIDATES=${CONTINUITY_CANDIDATES}"
echo "CONTINUITY_MIN_SCORE=${CONTINUITY_MIN_SCORE}"
echo "CONTINUITY_REGEN_ATTEMPTS=${CONTINUITY_REGEN_ATTEMPTS}"
echo "CRITIC_STORY_WEIGHT=${CRITIC_STORY_WEIGHT}"
echo "EMBEDDING_ADAPTER_CKPT=${EMBEDDING_ADAPTER_CKPT}"
echo "RUN_PRESET=${RUN_PRESET}"
echo "WINDOW_SHARD_COUNT=${WINDOW_SHARD_COUNT}"
echo "WINDOW_SHARD_INDEX=${WINDOW_SHARD_INDEX}"
echo "PARALLEL_WINDOW_MODE=${PARALLEL_WINDOW_MODE}"
echo "PARALLEL_GPUS_PER_NODE=${PARALLEL_GPUS_PER_NODE}"
echo "COMBINE_WINDOWS=${COMBINE_WINDOWS}"
echo "RUN_REPAIR_PASS=${RUN_REPAIR_PASS}"
echo "REPAIR_REPLACE_ORIGINAL=${REPAIR_REPLACE_ORIGINAL}"
echo "REPAIR_CANDIDATES=${REPAIR_CANDIDATES}"
echo "REPAIR_ATTEMPTS=${REPAIR_ATTEMPTS}"
echo "REPAIR_ACCEPT_SCORE=${REPAIR_ACCEPT_SCORE}"
echo "REPAIR_TRANSITION_THRESHOLD=${REPAIR_TRANSITION_THRESHOLD}"
echo "REPAIR_CRITIC_THRESHOLD=${REPAIR_CRITIC_THRESHOLD}"
echo "CAPTIONER_MODEL_ID=${CAPTIONER_MODEL_ID}"
echo "CAPTIONER_DEVICE=${CAPTIONER_DEVICE}"
=======
ARGS=(
--storyline "$storyline"
--window_plan_json "$window_plan_json"
--total_minutes $total_minutes
--window_seconds $window_seconds
--video_model_id "$wan_model"
--director_model_id "$director_model"
--embedding_backend dinov2
--embedding_model_id "$dinov2_model"
--captioner_model_id none
--num_frames $num_frames
--steps $steps
--guidance_scale $guidance_scale
--height $height
--width $width
--fps $fps
--seed $seed
--device $device
--output_dir "$output_dir"
--style_prefix "$style_prefix"
--negative_prompt "$negative_prompt"
--character_lock "$character_lock"
--reference_conditioning
--reference_tail_frames $reference_tail_frames
--reference_strength $reference_strength
--visual_quality_weight $visual_quality_weight
--initial_condition_image "$initial_image"
--continuity_candidates $continuity_candidates
--continuity_min_score $continuity_min_score
--continuity_regen_attempts $continuity_regen_attempts
--critic_story_weight $critic_story_weight
--transition_weight $transition_weight
--environment_weight $environment_weight
--disable_random_generation
--last_frame_memory
--shot_plan_defaults cinematic
--no-noise_conditioning
)

################################
# run SceneWeaver
################################

echo "===================================="
echo "Starting SceneWeaver pipeline"
echo "===================================="
>>>>>>> Stashed changes

echo "[DEBUG] Running command:"
printf '%q ' "$python_bin" scripts/run_story_pipeline.py "${ARGS[@]}"
echo ""

"$python_bin" scripts/run_story_pipeline.py "${ARGS[@]}"

################################
# job finished
################################

echo "===================================="
echo "SceneWeaver generation finished"
echo "End time: $(date)"
echo "===================================="

echo "[INFO] Generated clips:"
find "$output_dir" -name "*.mp4" || true

echo "===================================="
echo "Output directory:"
echo "$output_dir"
echo "===================================="
