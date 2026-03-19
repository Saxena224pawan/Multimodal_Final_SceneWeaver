#!/bin/bash -l
#SBATCH --job-name=sceneweaver_crow
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_root}"

mkdir -p slurm_logs outputs .hf

hpc_vault_root="${HPC_VAULT_ROOT:-/home/vault/v123be/v123be37}"
sceneweaver_runs_root="${SCENEWEAVER_RUNS_ROOT:-${hpc_vault_root}/sceneweaver_runs}"
mkdir -p "${sceneweaver_runs_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${project_root}/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

pipeline_mode="${PIPELINE_MODE:-core}"
wan_model="${WAN_MODEL:-${hpc_vault_root}/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers}"
director_model="${DIRECTOR_MODEL:-${hpc_vault_root}/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct}"
dinov2_model="${DINOV2_MODEL:-${hpc_vault_root}/facebook/dinov2-base}"
window_plan_json="${WINDOW_PLAN_JSON:-${project_root}/configs/window_plans/thirsty_crow_story.json}"

[ -d "$wan_model" ] || { echo "ERROR: Wan model missing at $wan_model"; exit 1; }
[ -d "$director_model" ] || { echo "ERROR: Director model missing at $director_model"; exit 1; }
[ -d "$dinov2_model" ] || { echo "ERROR: DINOv2 model missing at $dinov2_model"; exit 1; }
[ -f "$window_plan_json" ] || { echo "ERROR: Window plan missing at $window_plan_json"; exit 1; }

source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate sceneweaver311
python_bin="$(which python)"

total_minutes="${TOTAL_MINUTES:-0.8}"
window_seconds="${WINDOW_SECONDS:-8}"
fps="${FPS:-8}"
num_frames="${NUM_FRAMES:-65}"
steps="${STEPS:-28}"
guidance_scale="${GUIDANCE_SCALE:-5.4}"
height="${HEIGHT:-480}"
width="${WIDTH:-832}"
seed="${SEED:-42}"
device="${DEVICE:-cuda}"

continuity_candidates="${CONTINUITY_CANDIDATES:-2}"
continuity_min_score="${CONTINUITY_MIN_SCORE:-0.74}"
continuity_regen_attempts="${CONTINUITY_REGEN_ATTEMPTS:-2}"
critic_story_weight="${CRITIC_STORY_WEIGHT:-0.70}"
transition_weight="${TRANSITION_WEIGHT:-0.65}"
environment_weight="${ENVIRONMENT_WEIGHT:-0.35}"
visual_quality_weight="${VISUAL_QUALITY_WEIGHT:-0.18}"
style_weight="${STYLE_WEIGHT:-0.22}"

agent_max_iterations="${AGENT_MAX_ITERATIONS:-3}"
agent_quality_threshold="${AGENT_QUALITY_THRESHOLD:-0.76}"
agent_num_test_stories="${AGENT_NUM_TEST_STORIES:-}"

storyline="${STORYLINE:-A thirsty crow searches for water, finds a clay pot, raises the water by dropping stones into it, and finally drinks.}"
style_prefix="${STYLE_PREFIX:-cinematic storybook realism, single black crow, expressive eyes, detailed feathers, clear beak actions, stable camera, coherent motion, detailed clay pot, visible stones, warm daylight, high detail}"
negative_prompt="${NEGATIVE_PROMPT:-blurry, flicker, duplicate bird, extra birds, multiple crows, deformed beak, broken pot, background change, watermark, text, logo, collage, humans}"
character_lock="${CHARACTER_LOCK:-keep one black crow consistent across all windows, with the same clay pot and the same dusty courtyard under the same tree.}"

reference_strength="${REFERENCE_STRENGTH:-0.68}"
reference_tail_frames="${REFERENCE_TAIL_FRAMES:-4}"
initial_image="${INITIAL_IMAGE:-${project_root}/thirsty_crow_start_image.png}"
run_name_prefix="${RUN_NAME_PREFIX:-thirsty_crow_${pipeline_mode}}"
output_dir="${OUTPUT_DIR:-${sceneweaver_runs_root}/${run_name_prefix}_$(date +%y%m%d_%H%M%S)}"

if [ ! -f "$initial_image" ]; then
  "$python_bin" <<'PYEOF'
from PIL import Image, ImageDraw
img = Image.new('RGB', (832, 480), (214, 188, 140))
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
img.save('thirsty_crow_start_image.png')
PYEOF
fi

mkdir -p "$output_dir"

echo "SceneWeaver pipeline mode: $pipeline_mode"
echo "Output dir: $output_dir"

if [ "$pipeline_mode" = "agents" ]; then
  ARGS=(
    --storyline "$storyline"
    --window-plan-json "$window_plan_json"
    --output-dir "$output_dir"
    --video-model-id "$wan_model"
    --director-model-id "$director_model"
    --embedder-backend dinov2
    --embedding-model-id "$dinov2_model"
    --captioner-model-id __stub__
    --captioner-device cpu
    --max-iterations "$agent_max_iterations"
    --quality-threshold "$agent_quality_threshold"
    --total-minutes "$total_minutes"
    --window-seconds "$window_seconds"
    --num-frames "$num_frames"
    --fps "$fps"
    --steps "$steps"
    --guidance-scale "$guidance_scale"
    --height "$height"
    --width "$width"
    --seed "$seed"
    --device "$device"
    --reference-conditioning
    --reference-tail-frames "$reference_tail_frames"
    --reference-strength "$reference_strength"
    --disable-random-generation
    --initial-condition-image "$initial_image"
    --style-prefix "$style_prefix"
    --negative-prompt "$negative_prompt"
    --character-lock "$character_lock"
    --shot-plan-defaults cinematic
    --no-noise-conditioning
  )

  if [ -n "$agent_num_test_stories" ]; then
    ARGS+=(--num-test-stories "$agent_num_test_stories")
  fi

  printf '%q ' "$python_bin" scripts/run_story_pipeline_with_agents.py "${ARGS[@]}"
  echo
  "$python_bin" scripts/run_story_pipeline_with_agents.py "${ARGS[@]}"
elif [ "$pipeline_mode" = "core" ]; then
  ARGS=(
    --storyline "$storyline"
    --window_plan_json "$window_plan_json"
    --total_minutes "$total_minutes"
    --window_seconds "$window_seconds"
    --video_model_id "$wan_model"
    --director_model_id "$director_model"
    --embedding_backend dinov2
    --embedding_model_id "$dinov2_model"
    --captioner_model_id none
    --num_frames "$num_frames"
    --steps "$steps"
    --guidance_scale "$guidance_scale"
    --height "$height"
    --width "$width"
    --fps "$fps"
    --seed "$seed"
    --device "$device"
    --output_dir "$output_dir"
    --style_prefix "$style_prefix"
    --negative_prompt "$negative_prompt"
    --character_lock "$character_lock"
    --reference_conditioning
    --reference_tail_frames "$reference_tail_frames"
    --reference_strength "$reference_strength"
    --visual_quality_weight "$visual_quality_weight"
    --style_weight "$style_weight"
    --initial_condition_image "$initial_image"
    --continuity_candidates "$continuity_candidates"
    --continuity_min_score "$continuity_min_score"
    --continuity_regen_attempts "$continuity_regen_attempts"
    --critic_story_weight "$critic_story_weight"
    --transition_weight "$transition_weight"
    --environment_weight "$environment_weight"
    --disable_random_generation
    --last_frame_memory
    --shot_plan_defaults cinematic
    --no-noise_conditioning
  )

  printf '%q ' "$python_bin" scripts/run_story_pipeline.py "${ARGS[@]}"
  echo
  "$python_bin" scripts/run_story_pipeline.py "${ARGS[@]}"
else
  echo "ERROR: Unsupported PIPELINE_MODE '$pipeline_mode'. Use 'agents' or 'core'."
  exit 1
fi
