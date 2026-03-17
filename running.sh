#!/usr/bin/env bash
set -euo pipefail

echo "=== Starting sceneweaver_full job ==="
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Submit directory: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Start time: $(date)"
echo "--------------------------------------"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "${PROJECT_ROOT}"

# Create log folder if needed
mkdir -p slurm_logs

# (Optional) Activate conda / venv if you use one
# source ~/.bashrc
# conda activate your_env_name

# Print GPU info
echo "=== GPU Info ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; continuing without GPU inventory"
fi
echo "--------------------------------------"

# Run the active SceneWeaver pipeline entry point.
python scripts/run_story_pipeline.py "$@"

echo "--------------------------------------"
echo "Job finished at: $(date)"