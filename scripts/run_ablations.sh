#!/bin/bash -l
#SBATCH --job-name=sceneweaver_ablate
#SBATCH --output=slurm_logs/sceneweaver_ablation_%j.out
#SBATCH --error=slurm_logs/sceneweaver_ablation_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${project_root}"

mkdir -p slurm_logs outputs .hf

hpc_vault_root="${HPC_VAULT_ROOT:-/home/vault/v123be/v123be36}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HF_HOME="${project_root}/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

conda_sh="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
env_name="${ENV_NAME:-sceneweaver_runtime}"
source "${conda_sh}"
conda activate "${env_name}"
python_bin="$(which python)"

story_file="${1:-data/stories/story_01.txt}"
output_base="${OUTPUT_BASE:-outputs/ablations}"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
run_root="${output_base}/${timestamp}_${SLURM_JOB_ID:-local}"

wan_model="${WAN_MODEL:-${hpc_vault_root}/sceneweaver_models/Wan2.2-I2V-A14B-Diffusers}"
director_model="${DIRECTOR_MODEL:-${hpc_vault_root}/Multimodal_Final_SceneWeaver/LLM_MODEL/Qwen2.5-3B-Instruct}"
dinov2_model="${DINOV2_MODEL:-${hpc_vault_root}/facebook/dinov2-base}"
captioner_model="${CAPTIONER_MODEL:-__stub__}"
captioner_device="${CAPTIONER_DEVICE:-cpu}"
initial_image="${INITIAL_IMAGE:-${project_root}/default_start_image.png}"
window_plan_json="${WINDOW_PLAN_JSON:-}"

generic_style_prefix="cinematic storybook realism, expressive character acting, stable camera, coherent motion, readable staging, detailed environments, high detail"
generic_negative_prompt="blurry, flicker, duplicate subjects, deformed anatomy, broken perspective, background drift, watermark, text, logo, collage"
style_prefix="${STYLE_PREFIX:-${generic_style_prefix}}"
negative_prompt="${NEGATIVE_PROMPT:-${generic_negative_prompt}}"
character_lock="${CHARACTER_LOCK:-}"

disable_random_generation="${DISABLE_RANDOM_GENERATION:-0}"
max_iterations="${AGENT_MAX_ITERATIONS:-3}"
quality_threshold="${AGENT_QUALITY_THRESHOLD:-0.70}"
total_minutes="${TOTAL_MINUTES:-0.8}"
window_seconds="${WINDOW_SECONDS:-8}"
num_frames="${NUM_FRAMES:-65}"
fps="${FPS:-8}"
steps="${STEPS:-28}"
guidance_scale="${GUIDANCE_SCALE:-5.4}"
height="${HEIGHT:-480}"
width="${WIDTH:-832}"
seed="${SEED:-42}"
device="${DEVICE:-cuda}"
reference_tail_frames="${REFERENCE_TAIL_FRAMES:-4}"
reference_strength="${REFERENCE_STRENGTH:-0.68}"
num_test_stories="${AGENT_NUM_TEST_STORIES:-}"

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

require_path() {
    local label="$1"
    local target="$2"
    if [ ! -e "$target" ]; then
        echo "ERROR: ${label} missing at ${target}"
        exit 1
    fi
}

ensure_initial_image() {
    if [ -f "$initial_image" ]; then
        return
    fi

    "$python_bin" - <<'PYEOF'
from pathlib import Path
from PIL import Image, ImageDraw
path = Path("default_start_image.png")
img = Image.new("RGB", (832, 480), (128, 104, 78))
draw = ImageDraw.Draw(img)
draw.rectangle((0, 280, 832, 480), fill=(78, 94, 96))
draw.polygon([(0, 280), (180, 180), (320, 260), (470, 140), (650, 250), (832, 190), (832, 280)], fill=(110, 122, 126))
draw.rectangle((0, 0, 832, 190), fill=(172, 142, 112))
draw.ellipse((620, 45, 760, 165), fill=(233, 202, 124))
draw.rectangle((260, 180, 560, 320), outline=(214, 196, 164), width=6)
draw.rectangle((280, 205, 540, 315), fill=(194, 170, 138))
img.save(path)
PYEOF
}

print_header "Running Ablation Studies"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Story: $story_file"
echo "Output root: $run_root"
echo "Python: $python_bin"
echo "GPU visibility: ${CUDA_VISIBLE_DEVICES}"
if [ "$captioner_model" = "__stub__" ]; then
    echo "Captioner: stub mode"
else
    echo "Captioner: $captioner_model"
fi
echo ""

require_path "Story file" "$story_file"
require_path "Wan model" "$wan_model"
require_path "Director model" "$director_model"
require_path "DINOv2 model" "$dinov2_model"
ensure_initial_image
require_path "Initial condition image" "$initial_image"

mkdir -p "$run_root"

run_experiment() {
    local name=$1
    shift || true
    local output_dir="$run_root/$name"
    local extra_args=("$@")
    local cmd=(
        "$python_bin" scripts/run_story_pipeline_with_agents.py
        --storyline "$story_file"
        --output-dir "$output_dir"
        --video-model-id "$wan_model"
        --director-model-id "$director_model"
        --embedder-backend dinov2
        --embedding-model-id "$dinov2_model"
        --captioner-model-id "$captioner_model"
        --captioner-device "$captioner_device"
        --max-iterations "$max_iterations"
        --quality-threshold "$quality_threshold"
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
        --initial-condition-image "$initial_image"
        --style-prefix "$style_prefix"
        --negative-prompt "$negative_prompt"
        --character-lock "$character_lock"
        --shot-plan-defaults cinematic
        --no-noise-conditioning
    )

    if [ "$disable_random_generation" = "1" ]; then
        cmd+=(--disable-random-generation)
    fi

    if [ -n "$window_plan_json" ]; then
        require_path "Window plan JSON" "$window_plan_json"
        cmd+=(--window-plan-json "$window_plan_json")
    fi

    if [ -n "$num_test_stories" ]; then
        cmd+=(--num-test-stories "$num_test_stories")
    fi

    if [ ${#extra_args[@]} -gt 0 ]; then
        cmd+=("${extra_args[@]}")
    fi

    echo ""
    echo ">>> Running: $name"
    echo "Output: $output_dir"
    if [ ${#extra_args[@]} -gt 0 ]; then
        echo "Flags: ${extra_args[*]}"
    else
        echo "Flags: <baseline>"
    fi

    mkdir -p "$output_dir"
    printf '%s
' "${extra_args[*]:-baseline}" > "$output_dir/config.txt"

    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"

    "$python_bin" evaluation.py "$output_dir" > "$output_dir/eval.log" 2>&1
    echo "$name complete"
}

run_experiment "baseline_all_agents"
run_experiment "ablation_no_continuity" --disable-continuity
run_experiment "ablation_no_storybeats" --disable-storybeats
run_experiment "ablation_no_physics" --disable-physics
run_experiment "only_continuity" --only-continuity
run_experiment "only_storybeats" --only-storybeats
run_experiment "only_physics" --only-physics

print_header "ABLATION RESULTS COMPARISON"
printf '%-28s | %9s | %14s | %10s
' "Experiment" "Avg Score" "Avg Iterations" "Improvement"
printf '%-28s-+-%9s-+-%14s-+-%10s
' "----------------------------" "---------" "--------------" "----------"

for dir in "$run_root"/*/; do
    if [ -f "$dir/convergence_report.json" ]; then
        name=$(basename "$dir")
        avg_score=$(grep -o '"avg_final": [0-9.]*' "$dir/convergence_report.json" | cut -d: -f2 | tr -d ' ')
        avg_iter=$(grep -o '"avg_iterations_per_window": [0-9.]*' "$dir/convergence_report.json" | cut -d: -f2 | tr -d ' ')
        avg_impr=$(grep -o '"avg_improvement": [0-9.]*' "$dir/convergence_report.json" | cut -d: -f2 | tr -d ' ')
        printf '%-28s | %9.3f | %14.2f | %10.3f
' "$name" "$avg_score" "$avg_iter" "$avg_impr"
    fi
done

echo ""
echo "Results saved to: $run_root"
echo ""
echo "Next steps:"
echo "1. Review each convergence report under $run_root"
echo "2. Compare ablation_report.json files across runs"
echo "3. Aggregate repeated runs with python scripts/aggregate_results.py <parent_dir>"
