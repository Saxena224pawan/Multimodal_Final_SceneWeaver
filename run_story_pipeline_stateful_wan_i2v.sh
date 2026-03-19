#!/bin/bash -l
#SBATCH --job-name=sceneweaver_wan_i2v
#SBATCH --output=slurm_logs/sceneweaver_wan_i2v_%j.out
#SBATCH --error=slurm_logs/sceneweaver_wan_i2v_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
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

WAN_T2V_LOCAL_MODEL="${WAN_T2V_LOCAL_MODEL:-${SCENEWEAVER_WAN_MODEL_DIR}}"
T2V_MODEL_REPO="${T2V_MODEL_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
T2V_MODEL_DIR="${T2V_MODEL_DIR:-${PROJECT_ROOT}/models/$(basename "${T2V_MODEL_REPO}")}"
WAN_I2V_LOCAL_MODEL="${WAN_I2V_LOCAL_MODEL:-${SCENEWEAVER_WAN_I2V_MODEL_DIR}}"
I2V_MODEL_REPO="${I2V_MODEL_REPO:-Wan-AI/Wan2.1-I2V-14B-480P-Diffusers}"

TOTAL_MINUTES="${TOTAL_MINUTES:-1}"
WINDOW_SECONDS="${WINDOW_SECONDS:-8}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-12}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SEED="${SEED:-42}"
SEED_STRATEGY="${SEED_STRATEGY:-fixed}"
TAIL_FRAME_COUNT="${TAIL_FRAME_COUNT:-3}"
SAVE_CONDITIONING_FRAMES="${SAVE_CONDITIONING_FRAMES:-1}"

DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_ID:-}"
DIRECTOR_TEMPERATURE="${DIRECTOR_TEMPERATURE:-0.05}"
DIRECTOR_MAX_NEW_TOKENS="${DIRECTOR_MAX_NEW_TOKENS:-256}"
DIRECTOR_DO_SAMPLE="${DIRECTOR_DO_SAMPLE:-0}"
SHOT_PLAN_DEFAULTS="${SHOT_PLAN_DEFAULTS:-cinematic}"

DRY_RUN="${DRY_RUN:-0}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/story_run_stateful_wan_i2v_${RUN_STAMP}}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-scripts/run_story_pipeline_stateful_wan_i2v.py}"


die() {
  echo "$*" >&2
  exit 1
}


set_default_text() {
  local var_name="$1"
  local default_value="$2"

  if [ -z "${!var_name:-}" ]; then
    printf -v "${var_name}" '%s' "${default_value}"
  fi
}


prepare_workspace() {
  mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/.hf"
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
    export DIFFUSERS_OFFLINE=1
  fi
}


configure_prompt_defaults() {
  set_default_text "STORYLINE" "Asha hears footsteps in a narrow neon alley, turns left and tightens her grip on a red umbrella. She hides briefly under the rain and watches the alley entrance. She then runs toward the market exit while trying to stay unseen."
  set_default_text "STYLE_PREFIX" "cinematic realistic thriller, coherent motion, stable camera, grounded lighting, detailed textures"
  set_default_text "CHARACTER_LOCK" "keep the same named characters, face ids, wardrobe, and key props across windows; do not introduce extra people or unrelated objects"
  set_default_text "NEGATIVE_PROMPT" "blurry, low quality, flicker, frame jitter, deformed anatomy, duplicate subjects, extra limbs, extra people, crowd, wardrobe change, identity drift, wrong location, background swap, text, subtitles, watermark, logo, collage, split-screen, glitch"
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
  if [ -n "${WAN_T2V_LOCAL_MODEL}" ]; then
    T2V_MODEL_ID="${WAN_T2V_LOCAL_MODEL}"
  elif [ -d "${T2V_MODEL_DIR}" ]; then
    T2V_MODEL_ID="${T2V_MODEL_DIR}"
  else
    T2V_MODEL_ID="${T2V_MODEL_REPO}"
  fi

  if [ -n "${WAN_I2V_LOCAL_MODEL}" ] && [ -d "${WAN_I2V_LOCAL_MODEL}" ]; then
    I2V_MODEL_ID="${WAN_I2V_LOCAL_MODEL}"
  else
    I2V_MODEL_ID="${I2V_MODEL_REPO}"
  fi
}


validate_pipeline_script() {
  [ -f "${PIPELINE_SCRIPT}" ] || die "Pipeline script not found: ${PIPELINE_SCRIPT}"
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
    --total_minutes "${TOTAL_MINUTES}"
    --window_seconds "${WINDOW_SECONDS}"
    --t2v_model_id "${T2V_MODEL_ID}"
    --i2v_model_id "${I2V_MODEL_ID}"
    --director_temperature "${DIRECTOR_TEMPERATURE}"
    --director_max_new_tokens "${DIRECTOR_MAX_NEW_TOKENS}"
    --shot_plan_defaults "${SHOT_PLAN_DEFAULTS}"
    --style_prefix "${STYLE_PREFIX}"
    --character_lock "${CHARACTER_LOCK}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --num_frames "${NUM_FRAMES}"
    --steps "${STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --fps "${FPS}"
    --seed "${SEED}"
    --seed_strategy "${SEED_STRATEGY}"
    --device "${DEVICE}"
    --tail_frame_count "${TAIL_FRAME_COUNT}"
  )

  if [ -n "${DIRECTOR_MODEL_ID}" ]; then
    CMD+=(--director_model_id "${DIRECTOR_MODEL_ID}")
  fi
  if [ "${DIRECTOR_DO_SAMPLE}" = "1" ]; then
    CMD+=(--director_do_sample)
  else
    CMD+=(--no-director_do_sample)
  fi
  if [ "${SAVE_CONDITIONING_FRAMES}" = "1" ]; then
    CMD+=(--save_conditioning_frames)
  else
    CMD+=(--no-save_conditioning_frames)
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    CMD+=(--dry_run)
  fi
}


print_run_config() {
  sceneweaver_print_vars PROJECT_ROOT OUTPUT_DIR PIPELINE_SCRIPT PYTHON_BIN DEVICE T2V_MODEL_ID I2V_MODEL_ID DIRECTOR_MODEL_ID TAIL_FRAME_COUNT SAVE_CONDITIONING_FRAMES DRY_RUN
}


main() {
  load_modules_if_requested
  prepare_workspace
  activate_runtime
  configure_runtime_env
  configure_prompt_defaults
  resolve_python_bin
  resolve_model_ids
  validate_pipeline_script
  validate_device
  build_command
  print_run_config

  bash -n "${BASH_SOURCE[0]}"
  "${CMD[@]}"
}


main "$@"
