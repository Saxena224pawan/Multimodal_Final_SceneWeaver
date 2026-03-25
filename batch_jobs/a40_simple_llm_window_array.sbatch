#!/bin/bash -l
#SBATCH --job-name=sceneweaver_simple
#SBATCH --output=slurm_logs/sceneweaver_simple_%A_%a.out
#SBATCH --error=slurm_logs/sceneweaver_simple_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3%4

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_root}"

mkdir -p slurm_logs outputs .hf

hpc_vault_root="${HPC_VAULT_ROOT:-/home/vault/v123be/v123be36}"
sceneweaver_runs_root="${SCENEWEAVER_RUNS_ROOT:-${hpc_vault_root}/sceneweaver_runs}"
story_library_file="${STORY_LIBRARY_FILE:-${project_root}/configs/stories/short_stories.sh}"
conda_sh="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
env_name="${ENV_NAME:-sceneweaver_runtime}"

test -f "${story_library_file}" || { echo "ERROR: story library missing at ${story_library_file}"; exit 1; }
source "${story_library_file}"
source "${conda_sh}"
conda activate "${env_name}"

export PYTHONUNBUFFERED=1
export HF_HOME="${project_root}/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

story_keys=(
  "fox_and_grapes"
  "lion_and_mouse"
  "thirsty_crow"
  "tortoise_and_hare"
)

task_index="${SLURM_ARRAY_TASK_ID:-0}"
story_key="${story_keys[$task_index]}"
storyline="${SHORT_STORIES[$story_key]-}"
[ -n "${storyline}" ] || { echo "ERROR: story key '${story_key}' not found in ${story_library_file}"; exit 1; }

run_name_prefix="${RUN_NAME_PREFIX:-simple_${story_key}}"
output_dir="${OUTPUT_DIR:-${sceneweaver_runs_root}/${run_name_prefix}_$(date +%y%m%d_%H%M%S)}"
mkdir -p "${output_dir}"

python_bin="$(command -v python)"
director_model_id="${DIRECTOR_MODEL_ID:-${hpc_vault_root}/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct}"
video_model_id="${VIDEO_MODEL_ID:-${hpc_vault_root}/Wan2.2-T2V-A14B-Diffusers}"

[ -d "${director_model_id}" ] || { echo "ERROR: director model missing at ${director_model_id}"; exit 1; }
[ -d "${video_model_id}" ] || { echo "ERROR: video model missing at ${video_model_id}"; exit 1; }

echo "Simple SceneWeaver baseline"
echo "Story key: ${story_key}"
echo "Output dir: ${output_dir}"
echo "Python: ${python_bin}"
echo "Director model: ${director_model_id}"
echo "Video model: ${video_model_id}"
echo "Using default run_story_pipeline.py generation parameters"

"${python_bin}" scripts/run_story_pipeline.py   --storyline "${storyline}"   --output_dir "${output_dir}"   --director_model_id "${director_model_id}"   --video_model_id "${video_model_id}"   --window_count_mode fixed   --total_minutes 1.0   --window_seconds 10
