#!/bin/bash -l
#SBATCH --job-name=sceneweaver_crow
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16

set -euo pipefail

COMMON_SLURM_ROOT="${COMMON_SLURM_ROOT:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}}"
COMMON_SLURM_SH="${COMMON_SLURM_ROOT}/slurm_common.sh"
# shellcheck source=./slurm_common.sh
source "${COMMON_SLURM_SH}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-${SCENEWEAVER_DEFAULT_ENV}}"

CONDA_SH="${CONDA_SH:-${SCENEWEAVER_CONDA_SH}}"
USE_MODULES="${USE_MODULES:-0}"
PYTHON_MODULE="${PYTHON_MODULE:-${SCENEWEAVER_PYTHON_MODULE}}"
CUDA_MODULE="${CUDA_MODULE:-${SCENEWEAVER_CUDA_MODULE}}"
PYTHON_BIN="${PYTHON_BIN:-}"

USE_PROXY="${USE_PROXY:-1}"
PROXY_URL="${PROXY_URL:-${SCENEWEAVER_PROXY_URL}}"
NO_PROXY="${NO_PROXY:-${SCENEWEAVER_NO_PROXY}}"
USE_OFFLINE_MODE="${USE_OFFLINE_MODE:-1}"
STRICT_DEVICE="${STRICT_DEVICE:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

REFERENCE_CONDITIONING="${REFERENCE_CONDITIONING:-1}"
WAN_T2V_LOCAL_MODEL="${WAN_T2V_LOCAL_MODEL:-${SCENEWEAVER_WAN_MODEL_DIR}}"
WAN_T2V_MODEL_REPO="${WAN_T2V_MODEL_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
WAN_I2V_LOCAL_MODEL="${WAN_I2V_LOCAL_MODEL:-/home/vault/v123be/v123be36/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers}"
WAN_I2V_MODEL_REPO="${WAN_I2V_MODEL_REPO:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
DINOV2_LOCAL_MODEL="${DINOV2_LOCAL_MODEL:-${SCENEWEAVER_DINOV2_MODEL_DIR}}"
DINOV2_MODEL_REPO="${DINOV2_MODEL_REPO:-facebook/dinov2-base}"
DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_ID:-}"
WINDOW_PLAN_JSON="${WINDOW_PLAN_JSON:-${PROJECT_ROOT}/configs/window_plans/thirsty_crow_story.json}"
MODEL_LINKS="${MODEL_LINKS:-${PROJECT_ROOT}/outputs/pipeline/model_links.json}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-scripts/run_story_pipeline.py}"

TOTAL_MINUTES="${TOTAL_MINUTES:-0.8}"
WINDOW_SECONDS="${WINDOW_SECONDS:-8}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-65}"
STEPS="${STEPS:-28}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.4}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SEED="${SEED:-42}"
SEED_STRATEGY="${SEED_STRATEGY:-fixed}"

CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES:-2}"
CONTINUITY_MIN_SCORE="${CONTINUITY_MIN_SCORE:-0.74}"
CONTINUITY_REGEN_ATTEMPTS="${CONTINUITY_REGEN_ATTEMPTS:-2}"
CRITIC_STORY_WEIGHT="${CRITIC_STORY_WEIGHT:-0.70}"
TRANSITION_WEIGHT="${TRANSITION_WEIGHT:-0.65}"
ENVIRONMENT_WEIGHT="${ENVIRONMENT_WEIGHT:-0.35}"
VISUAL_QUALITY_WEIGHT="${VISUAL_QUALITY_WEIGHT:-0.18}"
REFERENCE_STRENGTH="${REFERENCE_STRENGTH:-0.68}"
REFERENCE_TAIL_FRAMES="${REFERENCE_TAIL_FRAMES:-4}"
SHOT_PLAN_DEFAULTS="${SHOT_PLAN_DEFAULTS:-cinematic}"

RUNS_ROOT="${RUNS_ROOT:-${SCENEWEAVER_VAULT_ROOT}/sceneweaver_runs}"
RUN_STAMP="$(date +%y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/thirsty_crow_${RUN_STAMP}}"
INITIAL_IMAGE="${INITIAL_IMAGE:-${PROJECT_ROOT}/thirsty_crow_start_image.png}"
DRY_RUN="${DRY_RUN:-0}"

STORYLINE="${STORYLINE:-A thirsty crow searches for water, finds a clay pot, raises the water by dropping stones into it, and finally drinks.}"
STYLE_PREFIX="${STYLE_PREFIX:-cinematic storybook realism, single black crow, expressive eyes, detailed feathers, clear beak actions, stable camera, coherent motion, detailed clay pot, visible stones, warm daylight, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, flicker, duplicate bird, extra birds, multiple crows, deformed beak, broken pot, background change, watermark, text, logo, collage, humans}"
CHARACTER_LOCK="${CHARACTER_LOCK:-keep one black crow consistent across all windows, with the same clay pot and the same dusty courtyard under the same tree.}"


die() {
  echo "$*" >&2
  exit 1
}


prepare_workspace() {
  mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/.hf" "${RUNS_ROOT}"
  cd "${PROJECT_ROOT}"
}


load_modules_if_requested() {
  if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
    module purge || true
    [ -n "${PYTHON_MODULE}" ] && module load "${PYTHON_MODULE}" || true
    [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
  fi
}


activate_runtime() {
  if [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ]; then
    ENV_PATH="${DEFAULT_ENV_PATH}"
  fi

  if [ -n "${ENV_PATH}" ]; then
    if [ -f "${CONDA_SH}" ]; then
      # shellcheck disable=SC1090
      source "${CONDA_SH}"
    fi
    command -v conda >/dev/null 2>&1 || die "Conda command not found. Set CONDA_SH correctly or activate env before running."
    conda activate "${ENV_PATH}"
    return
  fi

  if [ -n "${VENV_PATH}" ]; then
    [ -f "${VENV_PATH}/bin/activate" ] || die "Virtualenv activate script not found at ${VENV_PATH}/bin/activate"
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
  fi
}


configure_runtime_env() {
  export PYTHONUNBUFFERED=1
  export PYTHONNOUSERSITE=1
  export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf}"

  if [ "${USE_PROXY}" = "1" ] && [ -n "${PROXY_URL}" ]; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export HTTP_PROXY="${PROXY_URL}"
    export HTTPS_PROXY="${PROXY_URL}"
    export no_proxy="${NO_PROXY}"
    export NO_PROXY="${NO_PROXY}"
  fi

  if [ "${USE_OFFLINE_MODE}" = "1" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export DIFFUSERS_OFFLINE=1
  fi
}


resolve_python_bin() {
  if [ -n "${PYTHON_BIN}" ]; then
    return
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    die "No python interpreter found."
  fi
}


resolve_model_ids() {
  if [ "${REFERENCE_CONDITIONING}" = "1" ]; then
    if [ -n "${WAN_I2V_LOCAL_MODEL}" ] && [ -d "${WAN_I2V_LOCAL_MODEL}" ]; then
      VIDEO_MODEL_ID="${WAN_I2V_LOCAL_MODEL}"
    else
      VIDEO_MODEL_ID="${WAN_I2V_MODEL_REPO}"
    fi
  else
    if [ -n "${WAN_T2V_LOCAL_MODEL}" ] && [ -d "${WAN_T2V_LOCAL_MODEL}" ]; then
      VIDEO_MODEL_ID="${WAN_T2V_LOCAL_MODEL}"
    else
      VIDEO_MODEL_ID="${WAN_T2V_MODEL_REPO}"
    fi
  fi

  if [ -n "${DINOV2_LOCAL_MODEL}" ] && [ -d "${DINOV2_LOCAL_MODEL}" ]; then
    EMBEDDING_MODEL_ID="${DINOV2_LOCAL_MODEL}"
  else
    EMBEDDING_MODEL_ID="${DINOV2_MODEL_REPO}"
  fi
}


validate_inputs() {
  [ -f "${PIPELINE_SCRIPT}" ] || die "Pipeline script not found: ${PIPELINE_SCRIPT}"
  [ -f "${WINDOW_PLAN_JSON}" ] || die "Window plan missing at ${WINDOW_PLAN_JSON}"

  if [ ! -f "${INITIAL_IMAGE}" ]; then
    echo "[INFO] generating thirsty crow start image..."
    "${PYTHON_BIN}" <<'PYEOF'
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

  [ -f "${INITIAL_IMAGE}" ] || die "Initial image missing at ${INITIAL_IMAGE}"

  if [ "${REFERENCE_CONDITIONING}" = "1" ]; then
    if [ "${VIDEO_MODEL_ID}" = "${WAN_I2V_MODEL_REPO}" ] && [ "${USE_OFFLINE_MODE}" = "1" ]; then
      die "Wan I2V local model not found at ${WAN_I2V_LOCAL_MODEL}. Either make the local model available or set USE_OFFLINE_MODE=0."
    fi
  else
    if [ "${VIDEO_MODEL_ID}" = "${WAN_T2V_MODEL_REPO}" ] && [ "${USE_OFFLINE_MODE}" = "1" ]; then
      die "Wan T2V local model not found at ${WAN_T2V_LOCAL_MODEL}. Either make the local model available or set USE_OFFLINE_MODE=0."
    fi
  fi
  if [ "${EMBEDDING_MODEL_ID}" = "${DINOV2_MODEL_REPO}" ] && [ "${USE_OFFLINE_MODE}" = "1" ]; then
    die "DINOv2 local model not found at ${DINOV2_LOCAL_MODEL}. Either make the local model available or set USE_OFFLINE_MODE=0."
  fi
}


validate_device() {
  if [ "${DRY_RUN}" = "0" ] && [ "${DEVICE}" = "cuda" ]; then
    if ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
      echo "DEVICE=cuda requested, but torch.cuda.is_available() is false in this environment."
      if [ "${STRICT_DEVICE}" = "1" ]; then
        exit 1
      fi
      echo "Falling back to DEVICE=auto."
      DEVICE="auto"
    fi
  fi
}


build_command() {
  CMD=("${PYTHON_BIN}" "${PIPELINE_SCRIPT}"
    --storyline "${STORYLINE}"
    --output_dir "${OUTPUT_DIR}"
    --model_links "${MODEL_LINKS}"
    --total_minutes "${TOTAL_MINUTES}"
    --window_seconds "${WINDOW_SECONDS}"
    --window_plan_json "${WINDOW_PLAN_JSON}"
    --video_model_id "${VIDEO_MODEL_ID}"
    --dtype "${DTYPE}"
    --device "${DEVICE}"
    --num_frames "${NUM_FRAMES}"
    --steps "${STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --fps "${FPS}"
    --seed "${SEED}"
    --seed_strategy "${SEED_STRATEGY}"
    --reference_tail_frames "${REFERENCE_TAIL_FRAMES}"
    --reference_strength "${REFERENCE_STRENGTH}"
    --initial_condition_image "${INITIAL_IMAGE}"
    --style_prefix "${STYLE_PREFIX}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --character_lock "${CHARACTER_LOCK}"
    --embedding_backend dinov2
    --embedding_model_id "${EMBEDDING_MODEL_ID}"
    --last_frame_memory
    --continuity_candidates "${CONTINUITY_CANDIDATES}"
    --continuity_min_score "${CONTINUITY_MIN_SCORE}"
    --continuity_regen_attempts "${CONTINUITY_REGEN_ATTEMPTS}"
    --critic_story_weight "${CRITIC_STORY_WEIGHT}"
    --transition_weight "${TRANSITION_WEIGHT}"
    --environment_weight "${ENVIRONMENT_WEIGHT}"
    --visual_quality_weight "${VISUAL_QUALITY_WEIGHT}"
    --disable_random_generation
    --captioner_model_id none
    --shot_plan_defaults "${SHOT_PLAN_DEFAULTS}"
  )

  if [ "${REFERENCE_CONDITIONING}" = "1" ]; then
    CMD+=(--reference_conditioning)
  fi
  if [ -n "${DIRECTOR_MODEL_ID}" ]; then
    CMD+=(--director_model_id "${DIRECTOR_MODEL_ID}")
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    CMD+=(--dry_run)
  fi
}


print_config() {
  echo "===================================="
  echo "SceneWeaver job started"
  echo "Node: $(hostname)"
  echo "Job ID: ${SLURM_JOB_ID:-manual}"
  echo "Start time: $(date)"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Output directory: ${OUTPUT_DIR}"
  echo "ENV_PATH=${ENV_PATH:-}"
  echo "VENV_PATH=${VENV_PATH:-}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "REFERENCE_CONDITIONING=${REFERENCE_CONDITIONING}"
  echo "VIDEO_MODEL_ID=${VIDEO_MODEL_ID}"
  echo "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}"
  echo "DIRECTOR_MODEL_ID=${DIRECTOR_MODEL_ID:-<heuristic/local default>}"
  echo "WINDOW_PLAN_JSON=${WINDOW_PLAN_JSON}"
  echo "INITIAL_IMAGE=${INITIAL_IMAGE}"
  echo "USE_OFFLINE_MODE=${USE_OFFLINE_MODE}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "===================================="
  echo "[DEBUG] Running command:"
  printf '%q ' "${CMD[@]}"
  printf '\n'
}


run_pipeline() {
  "${CMD[@]}"
}


print_summary() {
  echo "===================================="
  echo "SceneWeaver generation finished"
  echo "End time: $(date)"
  echo "===================================="
  echo "[INFO] Generated clips:"
  find "${OUTPUT_DIR}" -name "*.mp4" || true
  echo "===================================="
  echo "Output directory:"
  echo "${OUTPUT_DIR}"
  echo "===================================="
}


prepare_workspace
load_modules_if_requested
activate_runtime
configure_runtime_env
resolve_python_bin
resolve_model_ids
validate_inputs
validate_device
build_command
print_config
run_pipeline
print_summary
