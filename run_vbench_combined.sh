#!/bin/bash -l
#SBATCH --job-name=vbench_all
#SBATCH --output=slurm_logs/vbench_all_%j.out
#SBATCH --error=slurm_logs/vbench_all_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

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

UPGRADE_PIP="${UPGRADE_PIP:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_WINDOW_PROMPT="${RUN_WINDOW_PROMPT:-1}"
RUN_CONTINUITY="${RUN_CONTINUITY:-1}"
REQUIRE_WINDOW_PROMPT="${REQUIRE_WINDOW_PROMPT:-0}"
AUTO_MATCH_STORY_RUN="${AUTO_MATCH_STORY_RUN:-1}"

EVAL_TARGET="${EVAL_TARGET:-${VIDEOS_PATH:-}}"
WINDOW_VIDEOS_PATH="${WINDOW_VIDEOS_PATH:-}"
CONTINUITY_VIDEOS_PATH="${CONTINUITY_VIDEOS_PATH:-}"
STORY_RUN_DIR="${STORY_RUN_DIR:-}"
STORY_RUN_SEARCH_ROOT="${STORY_RUN_SEARCH_ROOT:-${PROJECT_ROOT}/outputs}"

REPORT_ROOT="${REPORT_ROOT:-${PROJECT_ROOT}/outputs/reports/vbench_all_metrics}"
WINDOW_REPORT_ROOT="${WINDOW_REPORT_ROOT:-${REPORT_ROOT}/videobench_window_prompt}"
CONTINUITY_REPORT_ROOT="${CONTINUITY_REPORT_ROOT:-${REPORT_ROOT}/vbench_continuity}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
STORY_SLUG="${STORY_SLUG:-}"
RUN_NAME_BASE="${RUN_NAME_BASE:-}"

VIDEOBENCH_BIN="${VIDEOBENCH_BIN:-videobench}"
VIDEOBENCH_PACKAGE="${VIDEOBENCH_PACKAGE:-videobench}"
INSTALL_VIDEOBENCH="${INSTALL_VIDEOBENCH:-1}"
VIDEOBENCH_EXTRA_PIP_PACKAGES="${VIDEOBENCH_EXTRA_PIP_PACKAGES:-}"
VIDEOBENCH_CONFIG_PATH="${VIDEOBENCH_CONFIG_PATH:-}"
VIDEOBENCH_DIMENSIONS="${VIDEOBENCH_DIMENSIONS:-video-text consistency,action,scene,object_class,color}"
WINDOW_PROMPT_SOURCE="${WINDOW_PROMPT_SOURCE:-auto}"
WINDOW_MODE="${WINDOW_MODE:-auto}"
WINDOW_MODEL_NAME="${WINDOW_MODEL_NAME:-}"
WINDOW_LINK_MODE="${WINDOW_LINK_MODE:-auto}"
VIDEOBENCH_EXTRA_ARGS="${VIDEOBENCH_EXTRA_ARGS:-}"
SKIP_MISSING_PROMPTS="${SKIP_MISSING_PROMPTS:-0}"

VBENCH_BIN="${VBENCH_BIN:-vbench}"
VBENCH_PACKAGE="${VBENCH_PACKAGE:-vbench}"
INSTALL_VBENCH="${INSTALL_VBENCH:-1}"
VBENCH_EXTRA_PIP_PACKAGES="${VBENCH_EXTRA_PIP_PACKAGES:-setuptools==80.9.0}"
VBENCH_DIMENSIONS="${VBENCH_DIMENSIONS:-subject_consistency,background_consistency,motion_smoothness,temporal_flickering}"
CONTINUITY_MODE="${CONTINUITY_MODE:-custom_input}"
SEQUENCE_MODE="${SEQUENCE_MODE:-concat_windows}"
CHUNK_SECONDS="${CHUNK_SECONDS:-}"
CHUNK_FRAMES="${CHUNK_FRAMES:-}"
SEQUENCE_FALLBACK_PIP_PACKAGES="${SEQUENCE_FALLBACK_PIP_PACKAGES:-imageio imageio-ffmpeg}"
NGPUS="${NGPUS:-1}"
VBENCH_EXTRA_ARGS="${VBENCH_EXTRA_ARGS:-}"

trim() {
  local raw="${1:-}"
  raw="${raw#"${raw%%[![:space:]]*}"}"
  raw="${raw%"${raw##*[![:space:]]}"}"
  printf '%s' "${raw}"
}

slugify() {
  local raw
  raw="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  raw="$(printf '%s' "${raw}" | sed -E 's/\.[^.]+$//; s/_full_story$//; s/-full-story$//; s/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  if [ -z "${raw}" ]; then
    raw="story"
  fi
  printf '%s' "${raw}"
}

is_clips_dir() {
  [ -d "${1:-}" ] && [ "$(basename "${1}")" = "clips" ]
}

is_story_run_dir() {
  [ -d "${1:-}" ] && [ -d "${1}/clips" ]
}

upgrade_pip_once() {
  if [ "${UPGRADE_PIP}" = "1" ] && [ "${PIP_UPGRADED:-0}" = "0" ]; then
    "${PYTHON_BIN}" -m pip install --upgrade pip
    PIP_UPGRADED=1
  fi
}

ensure_python() {
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
}

install_if_missing() {
  local bin_name="$1"
  local install_flag="$2"
  local package_name="$3"
  local extra_packages="${4:-}"

  if [ "${DRY_RUN}" = "0" ] && ! command -v "${bin_name}" >/dev/null 2>&1; then
    if [ "${install_flag}" = "1" ]; then
      echo "Executable not found: ${bin_name}"
      echo "Attempting install with package: ${package_name}"
      upgrade_pip_once
      "${PYTHON_BIN}" -m pip install --upgrade "${package_name}"
      if [ -n "${extra_packages}" ]; then
        # shellcheck disable=SC2206
        local extra_arr=(${extra_packages})
        "${PYTHON_BIN}" -m pip install --upgrade "${extra_arr[@]}"
      fi
      hash -r
    fi
  fi

  if [ "${DRY_RUN}" = "0" ] && ! command -v "${bin_name}" >/dev/null 2>&1; then
    echo "Executable not found after install attempt: ${bin_name}"
    exit 1
  fi
}

ensure_pkg_resources() {
  if [ "${DRY_RUN}" = "0" ] && ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
    echo "pkg_resources missing; installing ${VBENCH_EXTRA_PIP_PACKAGES}"
    if [ -n "${VBENCH_EXTRA_PIP_PACKAGES}" ]; then
      # shellcheck disable=SC2206
      local extra_arr=(${VBENCH_EXTRA_PIP_PACKAGES})
      "${PYTHON_BIN}" -m pip install --upgrade "${extra_arr[@]}"
    fi
  fi

  if [ "${DRY_RUN}" = "0" ] && ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
    echo "pkg_resources still missing; forcing setuptools==80.9.0"
    "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "setuptools==80.9.0"
  fi

  if [ "${DRY_RUN}" = "0" ] && ! "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
    echo "ERROR: pkg_resources is still unavailable after fallback install."
    exit 1
  fi
}

ensure_sequence_fallbacks() {
  if [ "${DRY_RUN}" = "0" ] && [ "${SEQUENCE_MODE}" = "concat_windows" ] && ! command -v ffmpeg >/dev/null 2>&1; then
    if ! "${PYTHON_BIN}" -c "import imageio, imageio_ffmpeg" >/dev/null 2>&1; then
      echo "ffmpeg not found; installing Python concat fallback deps: ${SEQUENCE_FALLBACK_PIP_PACKAGES}"
      # shellcheck disable=SC2206
      local seq_arr=(${SEQUENCE_FALLBACK_PIP_PACKAGES})
      "${PYTHON_BIN}" -m pip install --upgrade "${seq_arr[@]}"
    fi
  fi

  if [ "${DRY_RUN}" = "0" ] && { [ -n "${CHUNK_SECONDS}" ] || [ -n "${CHUNK_FRAMES}" ]; }; then
    if ! "${PYTHON_BIN}" -c "import imageio, imageio_ffmpeg" >/dev/null 2>&1; then
      echo "Chunked evaluation requested; installing Python chunking deps: ${SEQUENCE_FALLBACK_PIP_PACKAGES}"
      # shellcheck disable=SC2206
      local seq_arr=(${SEQUENCE_FALLBACK_PIP_PACKAGES})
      "${PYTHON_BIN}" -m pip install --upgrade "${seq_arr[@]}"
    fi
  fi
}

resolve_story_run_from_video() {
  local video_path="$1"

  if [ -n "${STORY_RUN_DIR}" ]; then
    printf '%s\n' "${STORY_RUN_DIR}"
    return 0
  fi

  if [ "${AUTO_MATCH_STORY_RUN}" != "1" ]; then
    return 1
  fi

  "${PYTHON_BIN}" - "${video_path}" "${STORY_RUN_SEARCH_ROOT}" <<'PY'
import json
import re
import sys
from pathlib import Path

video_path = Path(sys.argv[1]).resolve()
search_root = Path(sys.argv[2]).resolve()

stem = video_path.stem.lower()
stem = re.sub(r"(_full_story|-full-story)$", "", stem)
tokens = [tok for tok in re.findall(r"[a-z0-9]+", stem) if tok not in {"a", "an", "and", "the", "full", "story", "video"}]

best = None
for summary_path in sorted(search_root.glob("story_run*/run_summary.json")):
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    text = json.dumps(payload, ensure_ascii=False).lower()
    score = sum(1 for tok in tokens if tok in text)
    if score <= 0:
        continue
    run_dir = summary_path.parent
    mtime = run_dir.stat().st_mtime
    candidate = (score, mtime, run_dir)
    if best is None or candidate[:2] > best[:2]:
        best = candidate

if best is not None:
    print(best[2].as_posix())
PY
}

append_extra_args() {
  local raw_items="$1"
  shift
  local -n out_ref="$1"

  if [ -n "${raw_items}" ]; then
    # shellcheck disable=SC2206
    local tmp_arr=(${raw_items})
    for arg in "${tmp_arr[@]}"; do
      out_ref+=(--extra_arg "${arg}")
    done
  fi
}

if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge || true
  [ -n "${PYTHON_MODULE}" ] && module load "${PYTHON_MODULE}" || true
  [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${REPORT_ROOT}"
cd "${PROJECT_ROOT}"

if [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ]; then
  ENV_PATH="${DEFAULT_ENV_PATH}"
fi

if [ -n "${ENV_PATH}" ]; then
  if [ -f "${CONDA_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  fi
  command -v conda >/dev/null 2>&1 || { echo "Conda command not found. Set CONDA_SH correctly or activate env before running."; exit 1; }
  conda activate "${ENV_PATH}"
elif [ -n "${VENV_PATH}" ]; then
  [ -f "${VENV_PATH}/bin/activate" ] || { echo "Virtualenv activate script not found at ${VENV_PATH}/bin/activate"; exit 1; }
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

if [ "${USE_PROXY}" = "1" ] && [ -n "${PROXY_URL}" ]; then
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  export no_proxy="${NO_PROXY}"
  export NO_PROXY="${NO_PROXY}"
fi

ensure_python
bash -n "${BASH_SOURCE[0]}"

TARGET_LABEL="<auto>"
RESOLVED_WINDOW_INPUT=""
RESOLVED_CONTINUITY_INPUT=""

if [ -n "${EVAL_TARGET}" ]; then
  EVAL_TARGET="$(readlink -f "${EVAL_TARGET}")"
  if [ ! -e "${EVAL_TARGET}" ]; then
    echo "EVAL_TARGET does not exist: ${EVAL_TARGET}"
    exit 1
  fi
  TARGET_LABEL="${EVAL_TARGET}"
fi

if [ -n "${WINDOW_VIDEOS_PATH}" ]; then
  RESOLVED_WINDOW_INPUT="$(readlink -f "${WINDOW_VIDEOS_PATH}")"
elif [ -n "${EVAL_TARGET}" ]; then
  if is_clips_dir "${EVAL_TARGET}" || is_story_run_dir "${EVAL_TARGET}"; then
    RESOLVED_WINDOW_INPUT="${EVAL_TARGET}"
  elif [ -f "${EVAL_TARGET}" ]; then
    RESOLVED_WINDOW_INPUT="$(trim "$(resolve_story_run_from_video "${EVAL_TARGET}" || true)")"
  fi
fi

if [ -n "${CONTINUITY_VIDEOS_PATH}" ]; then
  RESOLVED_CONTINUITY_INPUT="$(readlink -f "${CONTINUITY_VIDEOS_PATH}")"
elif [ -n "${EVAL_TARGET}" ]; then
  if is_story_run_dir "${EVAL_TARGET}"; then
    RESOLVED_CONTINUITY_INPUT="${EVAL_TARGET}/clips"
  else
    RESOLVED_CONTINUITY_INPUT="${EVAL_TARGET}"
  fi
fi

if [ -z "${STORY_SLUG}" ]; then
  if [ -n "${EVAL_TARGET}" ]; then
    if [ -f "${EVAL_TARGET}" ]; then
      STORY_SLUG="$(slugify "$(basename "${EVAL_TARGET}")")"
    elif is_clips_dir "${EVAL_TARGET}"; then
      STORY_SLUG="$(slugify "$(basename "$(dirname "${EVAL_TARGET}")")")"
    else
      STORY_SLUG="$(slugify "$(basename "${EVAL_TARGET}")")"
    fi
  else
    STORY_SLUG="latest_story"
  fi
fi

if [ -z "${RUN_NAME_BASE}" ]; then
  RUN_NAME_BASE="${STORY_SLUG}_${RUN_STAMP}"
fi

WINDOW_RUN_NAME="${WINDOW_RUN_NAME:-videobench_window_prompt_${RUN_NAME_BASE}}"
CONTINUITY_RUN_NAME="${CONTINUITY_RUN_NAME:-vbench_continuity_${RUN_NAME_BASE}}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "EVAL_TARGET=${TARGET_LABEL}"
echo "STORY_SLUG=${STORY_SLUG}"
echo "RUN_NAME_BASE=${RUN_NAME_BASE}"
echo "RUN_WINDOW_PROMPT=${RUN_WINDOW_PROMPT}"
echo "RUN_CONTINUITY=${RUN_CONTINUITY}"
echo "REQUIRE_WINDOW_PROMPT=${REQUIRE_WINDOW_PROMPT}"
echo "DRY_RUN=${DRY_RUN}"
echo "WINDOW_INPUT=${RESOLVED_WINDOW_INPUT:-<auto/latest story run>}"
echo "CONTINUITY_INPUT=${RESOLVED_CONTINUITY_INPUT:-<auto/latest story run>}"
echo "REPORT_ROOT=${REPORT_ROOT}"

if [ "${RUN_WINDOW_PROMPT}" = "1" ]; then
  if [ -n "${RESOLVED_WINDOW_INPUT}" ] || [ -z "${EVAL_TARGET}" ]; then
    install_if_missing "${VIDEOBENCH_BIN}" "${INSTALL_VIDEOBENCH}" "${VIDEOBENCH_PACKAGE}" "${VIDEOBENCH_EXTRA_PIP_PACKAGES}"

    if [ "${DRY_RUN}" = "0" ] && [ -z "${VIDEOBENCH_CONFIG_PATH}" ]; then
      echo "VIDEOBENCH_CONFIG_PATH is required when running Video-Bench."
      exit 1
    fi

    if [ "${DRY_RUN}" = "0" ] && [ ! -f "${VIDEOBENCH_CONFIG_PATH}" ]; then
      echo "Video-Bench config path does not exist: ${VIDEOBENCH_CONFIG_PATH}"
      exit 1
    fi

    WINDOW_CMD=("${PYTHON_BIN}" "scripts/10_eval_videobench_window_prompt.py"
      --report_root "${WINDOW_REPORT_ROOT}"
      --run_name "${WINDOW_RUN_NAME}"
      --dimensions "${VIDEOBENCH_DIMENSIONS}"
      --prompt_source "${WINDOW_PROMPT_SOURCE}"
      --mode "${WINDOW_MODE}"
      --model_name "${WINDOW_MODEL_NAME}"
      --link_mode "${WINDOW_LINK_MODE}"
      --videobench_bin "${VIDEOBENCH_BIN}"
    )

    if [ -n "${RESOLVED_WINDOW_INPUT}" ]; then
      WINDOW_CMD+=(--videos_path "${RESOLVED_WINDOW_INPUT}")
    fi
    if [ -n "${VIDEOBENCH_CONFIG_PATH}" ]; then
      WINDOW_CMD+=(--config_path "${VIDEOBENCH_CONFIG_PATH}")
    fi
    if [ "${SKIP_MISSING_PROMPTS}" = "1" ]; then
      WINDOW_CMD+=(--skip_missing_prompts)
    fi
    if [ "${DRY_RUN}" = "1" ]; then
      WINDOW_CMD+=(--dry_run)
    fi
    append_extra_args "${VIDEOBENCH_EXTRA_ARGS}" WINDOW_CMD

    echo "VIDEOBENCH_DIMENSIONS=${VIDEOBENCH_DIMENSIONS}"
    echo "WINDOW_PROMPT_SOURCE=${WINDOW_PROMPT_SOURCE}"
    echo "WINDOW_MODE=${WINDOW_MODE}"
    echo "WINDOW_LINK_MODE=${WINDOW_LINK_MODE}"
    echo "VIDEOBENCH_CONFIG_PATH=${VIDEOBENCH_CONFIG_PATH:-}"
    printf 'WINDOW_COMMAND='
    printf '%q ' "${WINDOW_CMD[@]}"
    printf '\n'

    "${WINDOW_CMD[@]}"
  else
    echo "Video-Bench window-prompt skipped: no story_run/clips metadata could be resolved for ${EVAL_TARGET}."
    if [ "${REQUIRE_WINDOW_PROMPT}" = "1" ]; then
      exit 1
    fi
  fi
fi

if [ "${RUN_CONTINUITY}" = "1" ]; then
  install_if_missing "${VBENCH_BIN}" "${INSTALL_VBENCH}" "${VBENCH_PACKAGE}" "${VBENCH_EXTRA_PIP_PACKAGES}"
  ensure_pkg_resources
  ensure_sequence_fallbacks

  CONTINUITY_CMD=("${PYTHON_BIN}" "scripts/09_eval_vbench_continuity.py"
    --mode "${CONTINUITY_MODE}"
    --sequence_mode "${SEQUENCE_MODE}"
    --report_root "${CONTINUITY_REPORT_ROOT}"
    --run_name "${CONTINUITY_RUN_NAME}"
    --dimensions "${VBENCH_DIMENSIONS}"
    --vbench_bin "${VBENCH_BIN}"
    --ngpus "${NGPUS}"
  )

  if [ -n "${RESOLVED_CONTINUITY_INPUT}" ]; then
    CONTINUITY_CMD+=(--videos_path "${RESOLVED_CONTINUITY_INPUT}")
  fi
  if [ -n "${CHUNK_SECONDS}" ]; then
    CONTINUITY_CMD+=(--chunk_seconds "${CHUNK_SECONDS}")
  fi
  if [ -n "${CHUNK_FRAMES}" ]; then
    CONTINUITY_CMD+=(--chunk_frames "${CHUNK_FRAMES}")
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    CONTINUITY_CMD+=(--dry_run)
  fi
  append_extra_args "${VBENCH_EXTRA_ARGS}" CONTINUITY_CMD

  echo "VBENCH_DIMENSIONS=${VBENCH_DIMENSIONS}"
  echo "CONTINUITY_MODE=${CONTINUITY_MODE}"
  echo "SEQUENCE_MODE=${SEQUENCE_MODE}"
  echo "CHUNK_SECONDS=${CHUNK_SECONDS}"
  echo "CHUNK_FRAMES=${CHUNK_FRAMES}"
  echo "NGPUS=${NGPUS}"
  printf 'CONTINUITY_COMMAND='
  printf '%q ' "${CONTINUITY_CMD[@]}"
  printf '\n'

  "${CONTINUITY_CMD[@]}"
fi
