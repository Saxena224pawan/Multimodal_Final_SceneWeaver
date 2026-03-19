#!/bin/bash
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$project_root"

mkdir -p slurm_logs

array_spec="${ARRAY_SPEC:-1-6%6}"
time_limit="${TIME_LIMIT:-08:00:00}"
mail_user="${MAIL_USER:-}"

sbatch_args=(
  --array "$array_spec"
  --time "$time_limit"
)

if [ -n "$mail_user" ]; then
  sbatch_args+=(--mail-type FAIL,END --mail-user "$mail_user")
fi

echo "Submitting SceneWeaver 6-story multi-GPU array job"
echo "Array spec: $array_spec"
echo "Time limit: $time_limit"
if [ -n "$mail_user" ]; then
  echo "Mail user: $mail_user"
fi
if [ "${CANCEL_ARRAY_ON_FAIL:-0}" = "1" ]; then
  echo "CANCEL_ARRAY_ON_FAIL=1 enabled"
fi

sbatch "${sbatch_args[@]}" batch_jobs/a100_story_batch6_array.sbatch
