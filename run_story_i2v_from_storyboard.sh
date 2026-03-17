#!/bin/bash -l
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR=/absolute/path/to/a_completed_t2v_run}"
ANCHOR_OUTPUT_DIR="${ANCHOR_OUTPUT_DIR:-${RUN_DIR}/anchors}"
ANCHOR_MAP_OUTPUT="${ANCHOR_MAP_OUTPUT:-${RUN_DIR}/window_reference_images.json}"
FRAME_POSITION="${FRAME_POSITION:-first}"
CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-sceneweaver311}"

cd "${PROJECT_ROOT}"

if [ -f "${CONDA_SH}" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
fi
command -v conda >/dev/null 2>&1 || { echo "Conda command not found. Set CONDA_SH correctly."; exit 1; }
conda activate "${ENV_PATH}"

python scripts/04_video/03_extract_window_anchor_frames.py   --run-dir "${RUN_DIR}"   --output-dir "${ANCHOR_OUTPUT_DIR}"   --map-output "${ANCHOR_MAP_OUTPUT}"   --frame-position "${FRAME_POSITION}"

export WINDOW_REFERENCE_IMAGES_JSON="${ANCHOR_MAP_OUTPUT}"
./run_story_pipeline.sh
