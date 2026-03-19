#!/bin/bash -l
#SBATCH --job-name=sceneweaver_batch6
#SBATCH --output=slurm_logs/sceneweaver_batch6_%A_%a.out
#SBATCH --error=slurm_logs/sceneweaver_batch6_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-6%6

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$project_root"

mkdir -p slurm_logs outputs .hf

hpc_vault_root="${HPC_VAULT_ROOT:-/home/vault/v123be/v123be37}"
sceneweaver_runs_root="${SCENEWEAVER_RUNS_ROOT:-${hpc_vault_root}/sceneweaver_runs}"
batch_root="${OUTPUT_ROOT:-${sceneweaver_runs_root}/story_batch6_array_${SLURM_ARRAY_JOB_ID}}"

story_idx="$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")"
story_file="${STORY_FILE_OVERRIDE:-${project_root}/data/stories/story_${story_idx}.txt}"
initial_image="${INITIAL_IMAGE:-${project_root}/default_start_image.png}"

video_model="${VIDEO_MODEL_ID:-${hpc_vault_root}/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers}"
director_model="${DIRECTOR_MODEL_ID:-${hpc_vault_root}/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct}"
embedding_model="${EMBEDDING_MODEL_ID:-${hpc_vault_root}/facebook/dinov2-base}"
heartbeat_seconds="${HEARTBEAT_SECONDS:-300}"
cancel_array_on_fail="${CANCEL_ARRAY_ON_FAIL:-0}"
window_count_mode="${WINDOW_COUNT_MODE:-dynamic}"
target_words_per_window="${TARGET_WORDS_PER_WINDOW:-28}"
min_dynamic_windows="${MIN_DYNAMIC_WINDOWS:-1}"
max_dynamic_windows="${MAX_DYNAMIC_WINDOWS:-24}"
num_frames="${NUM_FRAMES:-65}"
fps="${FPS:-8}"
steps="${STEPS:-28}"
guidance_scale="${GUIDANCE_SCALE:-5.4}"
height="${HEIGHT:-480}"
width="${WIDTH:-832}"
device="${DEVICE:-cuda}"
reference_tail_frames="${REFERENCE_TAIL_FRAMES:-4}"
reference_strength="${REFERENCE_STRENGTH:-0.68}"
continuity_regen_attempts="${CONTINUITY_REGEN_ATTEMPTS:-6}"
continuity_min_score="${CONTINUITY_MIN_SCORE:-0.78}"
continuity_convergence_patience="${CONTINUITY_CONVERGENCE_PATIENCE:-2}"
continuity_convergence_tolerance="${CONTINUITY_CONVERGENCE_TOLERANCE:-0.01}"
tightening_strength="${TIGHTENING_STRENGTH:-1.0}"
shot_plan_defaults="${SHOT_PLAN_DEFAULTS:-cinematic}"
embedding_backend="${EMBEDDING_BACKEND:-dinov2}"
captioner_model_id="${CAPTIONER_MODEL_ID:-none}"

[ -f "$story_file" ] || { echo "ERROR: story file missing at $story_file"; exit 1; }
[ -f "$initial_image" ] || { echo "ERROR: initial image missing at $initial_image"; exit 1; }
[ -d "$video_model" ] || { echo "ERROR: video model missing at $video_model"; exit 1; }
[ -d "$director_model" ] || { echo "ERROR: director model missing at $director_model"; exit 1; }
[ -d "$embedding_model" ] || { echo "ERROR: embedding model missing at $embedding_model"; exit 1; }

source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate sceneweaver311

export PYTHONUNBUFFERED=1
export HF_HOME="${project_root}/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

storyline="$(python - "$story_file" <<'PYEOF'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(encoding="utf-8")
print(" ".join(text.split()))
PYEOF
)"

story_slug="story_${story_idx}"
output_dir="${batch_root}/${story_slug}"
mkdir -p "$output_dir"

task_log="${output_dir}/task.log"
status_running="${output_dir}/STATUS.running"
status_ok="${output_dir}/STATUS.ok"
status_failed="${output_dir}/STATUS.failed"

rm -f "$status_running" "$status_ok" "$status_failed"
touch "$status_running"
exec > >(tee -a "$task_log") 2>&1

cleanup() {
  exit_code="$1"
  rm -f "$status_running"
  if [ -n "${heartbeat_pid:-}" ]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [ "$exit_code" -eq 0 ]; then
    printf 'status=ok
finished_at=%s
story_file=%s
output_dir=%s
'       "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$story_file" "$output_dir" > "$status_ok"
  else
    printf 'status=failed
failed_at=%s
exit_code=%s
story_file=%s
output_dir=%s
'       "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$exit_code" "$story_file" "$output_dir" > "$status_failed"
    if [ "$cancel_array_on_fail" = "1" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
      echo "[failure] canceling remaining array tasks for job ${SLURM_ARRAY_JOB_ID}"
      scancel "${SLURM_ARRAY_JOB_ID}" || true
    fi
  fi
}
trap 'cleanup "$?"' EXIT

heartbeat() {
  while true; do
    echo "[heartbeat] $(date '+%Y-%m-%d %H:%M:%S %Z') job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID} story=${story_idx} still running"
    sleep "$heartbeat_seconds"
  done
}
heartbeat &
heartbeat_pid=$!

echo "SceneWeaver multi-GPU array task"
echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Story file: ${story_file}"
echo "Output dir: ${output_dir}"
echo "Task log: ${task_log}"
echo "Heartbeat seconds: ${heartbeat_seconds}"
echo "Cancel array on fail: ${cancel_array_on_fail}"
echo "Window count mode: ${window_count_mode}"
echo "Max dynamic windows: ${max_dynamic_windows}"
echo "Frames/steps/fps: ${num_frames}/${steps}/${fps}"
echo "Embedding backend: ${embedding_backend}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<slurm-managed>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "Starting pipeline at $(date '+%Y-%m-%d %H:%M:%S %Z')"
python scripts/run_story_pipeline.py   --storyline "$storyline"   --output_dir "$output_dir"   --video_model_id "$video_model"   --director_model_id "$director_model"   --embedding_backend "$embedding_backend"   --embedding_model_id "$embedding_model"   --captioner_model_id "$captioner_model_id"   --window_count_mode "$window_count_mode" \
  --target_words_per_window "$target_words_per_window" \
  --min_dynamic_windows "$min_dynamic_windows" \
  --max_dynamic_windows "$max_dynamic_windows"   --num_frames "$num_frames"   --fps "$fps"   --steps "$steps"   --guidance_scale "$guidance_scale"   --height "$height"   --width "$width"   --device "$device"   --reference_conditioning   --initial_condition_image "$initial_image"   --reference_tail_frames "$reference_tail_frames"   --reference_strength "$reference_strength"   --disable_random_generation   --last_frame_memory   --continuity_regen_attempts "$continuity_regen_attempts"   --continuity_min_score "$continuity_min_score"   --continuity_convergence_patience "$continuity_convergence_patience"   --continuity_convergence_tolerance "$continuity_convergence_tolerance"   --progressive_tightening   --tightening_strength "$tightening_strength"   --shot_plan_defaults "$shot_plan_defaults"

echo "Finished pipeline at $(date '+%Y-%m-%d %H:%M:%S %Z')"
