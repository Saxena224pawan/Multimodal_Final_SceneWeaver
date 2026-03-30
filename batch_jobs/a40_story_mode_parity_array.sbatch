#!/bin/bash -l
#SBATCH --job-name=sceneweaver_parity
#SBATCH --output=slurm_logs/sceneweaver_parity_%A_%a.out
#SBATCH --error=slurm_logs/sceneweaver_parity_%A_%a.err
#SBATCH --time=12:00:00
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
parity_runs_root="${PARITY_RUNS_ROOT:-${sceneweaver_runs_root}/screenweaver_runs_parity}"
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

mode="${MODE:-core}"
run_tag="${RUN_TAG:-parity_$(date +%Y%m%d_%H%M%S)}"
output_dir="${parity_runs_root}/${mode}_${story_key}_${run_tag}"
mkdir -p "${output_dir}"
frame_width="${FRAME_WIDTH:-640}"
frame_height="${FRAME_HEIGHT:-384}"
window_seconds="${WINDOW_SECONDS:-10}"
num_frames="${NUM_FRAMES:-49}"
num_steps="${NUM_STEPS:-30}"

python_bin="$(command -v python)"
director_model_id="${DIRECTOR_MODEL_ID:-${hpc_vault_root}/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct}"
i2v_video_model_id="${I2V_VIDEO_MODEL_ID:-${hpc_vault_root}/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers}"
t2v_video_model_id="${T2V_VIDEO_MODEL_ID:-${hpc_vault_root}/Wan2.2-T2V-A14B-Diffusers}"
dinov2_model_id="${DINOV2_MODEL_ID:-${hpc_vault_root}/facebook/dinov2-base}"

[ -d "${director_model_id}" ] || { echo "ERROR: director model missing at ${director_model_id}"; exit 1; }
[ -d "${t2v_video_model_id}" ] || { echo "ERROR: T2V video model missing at ${t2v_video_model_id}"; exit 1; }
[ -d "${i2v_video_model_id}" ] || { echo "ERROR: I2V video model missing at ${i2v_video_model_id}"; exit 1; }

echo "SceneWeaver parity rerun"
echo "Mode: ${mode}"
echo "Story key: ${story_key}"
echo "Run tag: ${run_tag}"
echo "Output dir: ${output_dir}"
echo "Python: ${python_bin}"
echo "Director model: ${director_model_id}"
echo "T2V video model: ${t2v_video_model_id}"
echo "I2V video model: ${i2v_video_model_id}"
echo "Resolution: ${frame_width}x${frame_height}"
echo "Window seconds: ${window_seconds}"
echo "Frames: ${num_frames}"
echo "Steps: ${num_steps}"

generate_story_bootstrap_image() {
  local story_slug="$1"
  local image_path="$2"

  if [ -f "${image_path}" ]; then
    return 0
  fi

  "${python_bin}" - "${story_slug}" "${image_path}" <<'PY'
import sys
from pathlib import Path

from PIL import Image, ImageDraw

story_key = sys.argv[1]
image_path = Path(sys.argv[2])
image_path.parent.mkdir(parents=True, exist_ok=True)

W, H = 832, 480
img = Image.new("RGB", (W, H), (228, 219, 198))
d = ImageDraw.Draw(img)

# Shared sky and ground.
d.rectangle((0, 0, W, int(H * 0.68)), fill=(202, 225, 245))
d.rectangle((0, int(H * 0.68), W, H), fill=(184, 162, 110))
d.ellipse((560, 30, 760, 210), fill=(247, 228, 167))

if story_key == "thirsty_crow":
    d.rectangle((0, 350, W, H), fill=(191, 155, 102))
    d.ellipse((500, 80, 740, 300), fill=(110, 120, 75))
    d.rectangle((615, 140, 645, 360), fill=(108, 85, 50))
    d.ellipse((270, 150, 460, 390), fill=(166, 100, 52))
    d.rectangle((315, 220, 415, 390), fill=(166, 100, 52))
    d.ellipse((290, 120, 440, 230), fill=(189, 121, 68))
    d.ellipse((322, 225, 408, 314), fill=(70, 110, 145))
    d.ellipse((120, 195, 240, 255), fill=(25, 25, 25))
    d.polygon([(228, 222), (278, 206), (236, 240)], fill=(210, 160, 70))
    d.polygon([(104, 220), (78, 184), (96, 242)], fill=(18, 18, 18))
    d.polygon([(160, 252), (146, 304), (184, 252)], fill=(18, 18, 18))
    d.polygon([(188, 252), (178, 304), (214, 254)], fill=(18, 18, 18))
    for x, y in ((510, 360), (548, 376), (585, 352), (620, 370), (658, 350), (694, 366)):
        d.ellipse((x, y, x + 18, y + 14), fill=(120, 112, 102))
elif story_key == "fox_and_grapes":
    d.rectangle((0, 340, W, H), fill=(167, 143, 88))
    d.line((110, 70, 165, 300), fill=(98, 69, 41), width=15)
    d.line((165, 300, 280, 180), fill=(98, 69, 41), width=12)
    d.line((278, 180, 510, 160), fill=(98, 69, 41), width=9)
    for x, y in ((515, 170), (548, 160), (580, 176), (602, 154), (625, 184), (560, 202)):
        d.ellipse((x, y, x + 28, y + 28), fill=(91, 55, 126))
    d.ellipse((170, 265, 360, 355), fill=(198, 112, 47))
    d.ellipse((315, 278, 392, 330), fill=(198, 112, 47))
    d.polygon([(382, 304), (438, 288), (392, 320)], fill=(242, 220, 185))
    d.polygon([(325, 272), (294, 214), (346, 255)], fill=(198, 112, 47))
    d.polygon([(350, 276), (336, 210), (388, 258)], fill=(198, 112, 47))
    d.polygon([(180, 320), (110, 360), (176, 342)], fill=(241, 226, 210))
    d.polygon([(124, 360), (96, 390), (150, 368)], fill=(198, 112, 47))
    d.polygon([(212, 350), (190, 410), (236, 356)], fill=(82, 53, 33))
    d.polygon([(258, 352), (246, 414), (288, 360)], fill=(82, 53, 33))
elif story_key == "lion_and_mouse":
    d.rectangle((0, 330, W, H), fill=(196, 176, 110))
    d.line((620, 90, 615, 340), fill=(112, 92, 62), width=18)
    d.line((615, 120, 520, 190), fill=(112, 92, 62), width=10)
    d.line((615, 150, 720, 220), fill=(112, 92, 62), width=10)
    d.line((615, 180, 545, 280), fill=(112, 92, 62), width=10)
    d.line((615, 205, 705, 300), fill=(112, 92, 62), width=10)
    d.ellipse((210, 200, 450, 340), fill=(181, 132, 72))
    d.ellipse((365, 170, 510, 300), fill=(200, 151, 88))
    d.ellipse((340, 150, 522, 332), outline=(120, 78, 36), width=18)
    d.polygon([(198, 262), (120, 220), (164, 286)], fill=(181, 132, 72))
    d.polygon([(430, 322), (520, 352), (596, 318), (640, 274)], fill=(120, 78, 36))
    d.polygon([(250, 336), (226, 422), (286, 344)], fill=(120, 78, 36))
    d.polygon([(320, 338), (304, 426), (356, 344)], fill=(120, 78, 36))
    d.ellipse((150, 330, 220, 366), fill=(142, 122, 98))
    d.ellipse((206, 332, 236, 352), fill=(142, 122, 98))
    d.arc((118, 326, 190, 392), 300, 70, fill=(142, 122, 98), width=4)
    for x in (522, 560, 598, 636, 674):
        d.line((x, 160, x, 335), fill=(164, 168, 152), width=3)
    for y in (168, 200, 232, 264, 296, 328):
        d.line((510, y, 694, y), fill=(164, 168, 152), width=3)
elif story_key == "tortoise_and_hare":
    d.rectangle((0, 330, W, H), fill=(170, 145, 96))
    d.polygon([(0, 368), (220, 334), (430, 318), (610, 340), (832, 314), (832, 408), (0, 440)], fill=(204, 184, 134))
    d.line((100, 130, 90, 320), fill=(105, 88, 60), width=16)
    d.ellipse((44, 72, 154, 166), fill=(108, 134, 78))
    d.ellipse((86, 244, 216, 324), fill=(198, 213, 188))
    d.ellipse((180, 254, 236, 302), fill=(198, 213, 188))
    d.polygon([(195, 248), (172, 198), (206, 234)], fill=(198, 213, 188))
    d.polygon([(220, 250), (220, 196), (246, 236)], fill=(198, 213, 188))
    d.polygon([(118, 312), (104, 370), (140, 318)], fill=(125, 98, 74))
    d.polygon([(164, 312), (158, 372), (192, 318)], fill=(125, 98, 74))
    d.ellipse((470, 292, 600, 354), fill=(94, 122, 76))
    d.ellipse((570, 302, 608, 330), fill=(94, 122, 76))
    d.ellipse((490, 278, 580, 348), outline=(71, 94, 59), width=10)
    d.polygon([(506, 346), (494, 392), (524, 348)], fill=(63, 78, 50))
    d.polygon([(548, 346), (544, 392), (572, 350)], fill=(63, 78, 50))
else:
    d.rectangle((0, 340, W, H), fill=(191, 155, 102))
    d.ellipse((200, 180, 420, 360), fill=(120, 120, 120))

img.save(image_path)
print(image_path)
PY
}

common_simple_args=(
  --storyline "${storyline}"
  --output_dir "${output_dir}"
  --director_model_id "${director_model_id}"
  --video_model_id "${t2v_video_model_id}"
  --window_count_mode fixed
  --total_minutes 1.0
  --window_seconds "${window_seconds}"
  --embedding_backend none
  --num_frames "${num_frames}"
  --steps "${num_steps}"
  --guidance_scale 6.0
  --fps 8
  --width "${frame_width}"
  --height "${frame_height}"
  --no-reference_conditioning
)

common_core_args=(
  "${common_simple_args[@]}"
)

common_agent_args=(
  --storyline "${storyline}"
  --output-dir "${output_dir}"
  --director-model-id "${director_model_id}"
  --video-model-id "${t2v_video_model_id}"
  --window-count-mode fixed
  --total-minutes 1.0
  --window-seconds "${window_seconds}"
  --embedder-backend none
  --num-frames "${num_frames}"
  --steps "${num_steps}"
  --guidance-scale 6.0
  --fps 8
  --width "${frame_width}"
  --height "${frame_height}"
  --quality-threshold "${AGENT_QUALITY_THRESHOLD:-0.76}"
  --max-iterations "${AGENT_MAX_ITERATIONS:-5}"
  --convergence-patience "${AGENT_CONVERGENCE_PATIENCE:-2}"
  --convergence-tolerance "${AGENT_CONVERGENCE_TOLERANCE:-0.015}"
  --style-prefix "cinematic realistic, coherent motion, stable camera, high detail"
  --character-lock "keep the same main characters, wardrobe, key props, and scene scale across windows; no duplicate subjects and no unrelated new objects"
  --negative-prompt "blurry, low quality, flicker, frame jitter, deformed anatomy, duplicate subjects, extra limbs, extra animals, wrong species, text, subtitles, watermark, logo, collage, split-screen, glitch"
  --no-reference-conditioning
)

bootstrap_image="${output_dir}/bootstrap_${story_key}.png"
simple_i2v_args=(
  --storyline "${storyline}"
  --output_dir "${output_dir}"
  --director_model_id "${director_model_id}"
  --video_model_id "${i2v_video_model_id}"
  --window_count_mode fixed
  --total_minutes 1.0
  --window_seconds "${window_seconds}"
  --embedding_backend none
  --num_frames "${num_frames}"
  --steps "${num_steps}"
  --guidance_scale 6.0
  --fps 8
  --width "${frame_width}"
  --height "${frame_height}"
  --reference_conditioning
  --initial_condition_image "${bootstrap_image}"
)

case "${mode}" in
  simple)
    printf '%q ' "${python_bin}" scripts/run_story_pipeline.py "${common_simple_args[@]}"
    printf '\n'
    exec "${python_bin}" scripts/run_story_pipeline.py "${common_simple_args[@]}"
    ;;
  core)
    printf '%q ' "${python_bin}" scripts/run_story_pipeline.py "${common_core_args[@]}"
    printf '\n'
    exec "${python_bin}" scripts/run_story_pipeline.py "${common_core_args[@]}"
    ;;
  agents|agentic)
    if [ -d "${dinov2_model_id}" ]; then
      echo "Parity run keeps embedder disabled for fairness; local DINOv2 remains available at ${dinov2_model_id}"
    fi
    printf '%q ' "${python_bin}" scripts/run_story_pipeline_with_agents.py "${common_agent_args[@]}"
    printf '\n'
    exec "${python_bin}" scripts/run_story_pipeline_with_agents.py "${common_agent_args[@]}"
    ;;
  simple_i2v|simple-i2v)
    generate_story_bootstrap_image "${story_key}" "${bootstrap_image}"
    printf '%q ' "${python_bin}" scripts/run_story_pipeline.py "${simple_i2v_args[@]}"
    printf '\n'
    exec "${python_bin}" scripts/run_story_pipeline.py "${simple_i2v_args[@]}"
    ;;
  *)
    echo "ERROR: unsupported MODE '${mode}'. Use 'simple', 'core', 'agents', or 'simple_i2v'."
    exit 1
    ;;
esac
