#!/bin/bash -l
#SBATCH --job-name=sceneweaver_pavan4
#SBATCH --output=slurm_logs/sceneweaver_pavan4_%A_%a.out
#SBATCH --error=slurm_logs/sceneweaver_pavan4_%A_%a.err
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3%4

set -euo pipefail

project_root="${PROJECT_ROOT:-/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver}"
cd "${project_root}"

mkdir -p slurm_logs outputs .hf

hpc_vault_root="${HPC_VAULT_ROOT:-/home/vault/v123be/v123be36}"
story_library_file="${STORY_LIBRARY_FILE:-${project_root}/configs/stories/short_stories.sh}"
output_root="${OUTPUT_ROOT:-${project_root}/outputs/story_runs_origin_pavan}"
concat_output_root="${CONCAT_OUTPUT_ROOT:-}"
run_tag="${RUN_TAG:-origin_pavan_$(date +%Y%m%d_%H%M%S)}"
pipeline_mode="${PIPELINE_MODE:-core}"
video_mode="${VIDEO_MODE:-t2v}"

t2v_video_model_id="${T2V_VIDEO_MODEL_ID:-${hpc_vault_root}/Wan2.2-T2V-A14B-Diffusers}"
i2v_video_model_id="${I2V_VIDEO_MODEL_ID:-${hpc_vault_root}/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers}"
director_model_id="${DIRECTOR_MODEL_ID:-${hpc_vault_root}/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct}"
embedding_model_id="${EMBEDDING_MODEL_ID:-${hpc_vault_root}/facebook/dinov2-base}"
embedding_adapter_ckpt="${EMBEDDING_ADAPTER_CKPT:-${project_root}/outputs/pororo_continuity_adapter.pt}"

conda_sh="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
env_name="${ENV_NAME:-sceneweaver_runtime}"

source "${story_library_file}"
story_keys=(
  "fox_and_grapes"
  "lion_and_mouse"
  "thirsty_crow"
  "tortoise_and_hare"
)

task_index="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_index}" -lt 0 ] || [ "${task_index}" -ge "${#story_keys[@]}" ]; then
  echo "Array index ${task_index} is out of range for ${#story_keys[@]} stories."
  exit 1
fi

selected_story="${story_keys[$task_index]}"
storyline="${SHORT_STORIES[$selected_story]-}"
[ -n "${storyline}" ] || { echo "ERROR: Story key '${selected_story}' not found in ${story_library_file}"; exit 1; }

[ -d "${t2v_video_model_id}" ] || { echo "ERROR: T2V video model missing at ${t2v_video_model_id}"; exit 1; }
[ -d "${i2v_video_model_id}" ] || { echo "ERROR: I2V video model missing at ${i2v_video_model_id}"; exit 1; }
[ -d "${director_model_id}" ] || { echo "ERROR: director model missing at ${director_model_id}"; exit 1; }
if [ "${pipeline_mode}" != "simple" ]; then
  [ -d "${embedding_model_id}" ] || { echo "ERROR: embedding model missing at ${embedding_model_id}"; exit 1; }
fi
if [ "${pipeline_mode}" = "core" ]; then
  [ -f "${embedding_adapter_ckpt}" ] || { echo "ERROR: embedding adapter checkpoint missing at ${embedding_adapter_ckpt}"; exit 1; }
fi

source "${conda_sh}"
conda activate "${env_name}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${project_root}/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python_bin="$(command -v python)"
mkdir -p "${output_root}"
output_dir="${output_root}/${pipeline_mode}_${selected_story}_${run_tag}"
concat_output_name="${CONCAT_OUTPUT_NAME:-${selected_story}_full_story.mp4}"
simple_window_reference_json="${output_dir}/simple_i2v_window_references.json"
concat_output_dir="${output_dir}"
concat_output_path="${concat_output_dir}/${concat_output_name}"
concat_work_dir="${concat_output_dir}/_concatenated_story"
if [ -n "${concat_output_root}" ]; then
  concat_output_dir="${concat_output_root}/${pipeline_mode}_${selected_story}_${run_tag}"
  concat_output_path="${concat_output_dir}/${concat_output_name}"
  concat_work_dir="${concat_output_dir}/_concatenated_story"
fi

window_seconds="${WINDOW_SECONDS:-10}"
window_count_mode="${WINDOW_COUNT_MODE:-fixed}"
fixed_windows="${FIXED_WINDOWS:-8}"
total_minutes="${TOTAL_MINUTES:-$(awk "BEGIN { printf \"%.6f\", (${fixed_windows} * ${window_seconds}) / 60.0 }")}"
continuity_candidates="${CONTINUITY_CANDIDATES:-2}"
continuity_regen_attempts="${CONTINUITY_REGEN_ATTEMPTS:-2}"
continuity_min_score="${CONTINUITY_MIN_SCORE:-0.72}"
reference_tail_frames="${REFERENCE_TAIL_FRAMES:-4}"
reference_strength="${REFERENCE_STRENGTH:-0.70}"
initial_condition_image="${INITIAL_CONDITION_IMAGE:-}"

num_frames="${NUM_FRAMES:-33}"
steps="${STEPS:-20}"
guidance_scale="${GUIDANCE_SCALE:-6.0}"
height="${HEIGHT:-384}"
width="${WIDTH:-640}"
fps="${FPS:-8}"
seed="${SEED:-42}"

echo "SceneWeaver origin/pavan-style run"
echo "Mode: ${pipeline_mode}"
echo "Video mode: ${video_mode}"
echo "Story: ${selected_story}"
echo "Output dir: ${output_dir}"
echo "T2V video model: ${t2v_video_model_id}"
echo "I2V video model: ${i2v_video_model_id}"
echo "Director model: ${director_model_id}"
echo "Embedding model: ${embedding_model_id}"
echo "Embedding adapter: ${embedding_adapter_ckpt}"
echo "Window planning mode: ${window_count_mode}"
echo "Target windows/story: ${fixed_windows}"
echo "Window seconds: ${window_seconds}"
echo "Target minutes/story: ${total_minutes}"
echo "Concatenated story output: ${concat_output_path}"

if [ -z "${initial_condition_image}" ]; then
  if [ "${selected_story}" = "thirsty_crow" ] && [ -f "${project_root}/thirsty_crow_start_image.png" ]; then
    initial_condition_image="${project_root}/thirsty_crow_start_image.png"
  elif [ -f "${project_root}/default_start_image.png" ]; then
    initial_condition_image="${project_root}/default_start_image.png"
  fi
fi

selected_video_model_id="${t2v_video_model_id}"
simple_core_reference_args=(--no-reference_conditioning)
agent_reference_args=(--no-reference-conditioning)
simple_extra_args=()
if [ "${video_mode}" = "i2v" ]; then
  selected_video_model_id="${i2v_video_model_id}"
  [ -n "${initial_condition_image}" ] || { echo "ERROR: VIDEO_MODE=i2v requires INITIAL_CONDITION_IMAGE or a repo default starter image."; exit 1; }
  [ -f "${initial_condition_image}" ] || { echo "ERROR: initial condition image missing at ${initial_condition_image}"; exit 1; }
  simple_core_reference_args=(
    --reference_conditioning
    --reference_tail_frames "${reference_tail_frames}"
    --reference_strength "${reference_strength}"
    --initial_condition_image "${initial_condition_image}"
  )
  agent_reference_args=(
    --reference-conditioning
    --reference-tail-frames "${reference_tail_frames}"
    --reference-strength "${reference_strength}"
    --initial-condition-image "${initial_condition_image}"
  )
fi

if [ "${video_mode}" = "i2v" ] && [ "${pipeline_mode}" = "simple" ]; then
  reference_map_windows="${fixed_windows}"
  if [ "${window_count_mode}" != "fixed" ]; then
    reference_map_windows="${MAX_DYNAMIC_WINDOWS:-24}"
  fi
  "${python_bin}" - "${simple_window_reference_json}" "${initial_condition_image}" "${reference_map_windows}" <<'PYREF'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
image_path = sys.argv[2]
window_count = max(1, int(sys.argv[3]))
payload = {}
for idx in range(window_count):
    payload[str(idx)] = [image_path]
    payload[f"w_{idx:03d}"] = [image_path]
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(out_path.as_posix())
PYREF
  simple_extra_args+=(--window_reference_images_json "${simple_window_reference_json}")
fi

echo "Selected video model: ${selected_video_model_id}"
echo "Reference tail frames: ${reference_tail_frames}"
echo "Reference strength: ${reference_strength}"
echo "Initial condition image: ${initial_condition_image:-<none>}"
echo "Simple window reference json: ${simple_window_reference_json:-<none>}"

case "${pipeline_mode}" in
  simple)
    cmd=(
      "${python_bin}" -s -u scripts/run_story_pipeline.py
      --storyline "${storyline}"
      --output_dir "${output_dir}"
      --total_minutes "${total_minutes}"
      --window_seconds "${window_seconds}"
      --window_count_mode "${window_count_mode}"
      --video_model_id "${selected_video_model_id}"
      --director_model_id "${director_model_id}"
      --device auto
      --embedding_backend none
      --num_frames "${num_frames}"
      --steps "${steps}"
      --guidance_scale "${guidance_scale}"
      --height "${height}"
      --width "${width}"
      --fps "${fps}"
      --seed "${seed}"
      --captioner_model_id none
      "${simple_core_reference_args[@]}"
      "${simple_extra_args[@]}"
    )
    ;;
  core)
    cmd=(
      "${python_bin}" -s -u scripts/run_story_pipeline.py
      --storyline "${storyline}"
      --output_dir "${output_dir}"
      --total_minutes "${total_minutes}"
      --window_seconds "${window_seconds}"
      --window_count_mode "${window_count_mode}"
      --video_model_id "${selected_video_model_id}"
      --director_model_id "${director_model_id}"
      --device auto
      --embedding_backend dinov2
      --embedding_model_id "${embedding_model_id}"
      --embedding_adapter_ckpt "${embedding_adapter_ckpt}"
      --last_frame_memory
      --continuity_candidates "${continuity_candidates}"
      --continuity_regen_attempts "${continuity_regen_attempts}"
      --continuity_min_score "${continuity_min_score}"
      --num_frames "${num_frames}"
      --steps "${steps}"
      --guidance_scale "${guidance_scale}"
      --height "${height}"
      --width "${width}"
      --fps "${fps}"
      --seed "${seed}"
      --captioner_model_id none
      "${simple_core_reference_args[@]}"
    )
    ;;
  agentic)
    cmd=(
      "${python_bin}" -s -u scripts/run_story_pipeline_with_agents.py
      --storyline "${storyline}"
      --output-dir "${output_dir}"
      --video-model-id "${selected_video_model_id}"
      --director-model-id "${director_model_id}"
      --embedder-backend dinov2
      --embedding-model-id "${embedding_model_id}"
      --captioner-model-id __stub__
      --captioner-device cpu
      --total-minutes "${total_minutes}"
      --window-seconds "${window_seconds}"
      --window-count-mode "${window_count_mode}"
      --num-frames "${num_frames}"
      --steps "${steps}"
      --guidance-scale "${guidance_scale}"
      --height "${height}"
      --width "${width}"
      --fps "${fps}"
      --seed "${seed}"
      --device auto
      --max-iterations 5
      --quality-threshold 0.76
      "${agent_reference_args[@]}"
    )
    ;;
  *)
    echo "ERROR: Unsupported PIPELINE_MODE '${pipeline_mode}'. Use simple, core, or agentic."
    exit 1
    ;;
esac

printf 'COMMAND='
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

concat_cmd=(
  "${python_bin}" -s -u scripts/11_concat_story_windows.py
  --run-dir "${output_dir}"
  --output-path "${concat_output_path}"
  --work-dir "${concat_work_dir}"
)
printf 'CONCAT_COMMAND='
printf '%q ' "${concat_cmd[@]}"
printf '\n'
"${concat_cmd[@]}"
