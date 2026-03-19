#!/bin/bash
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$project_root"

jobs=(
  "batch_jobs/a100_thirsty_crow_core.sbatch"
  "batch_jobs/a100_lion_mouse_core.sbatch"
  "batch_jobs/a100_fox_grapes_core.sbatch"
  "batch_jobs/a100_tortoise_hare_core.sbatch"
)

for job in "${jobs[@]}"; do
  echo "Submitting $job"
  sbatch "$job"
done
