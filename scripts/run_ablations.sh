#!/bin/bash
# Ablation Study Runner - Test individual agent contributions
# Usage: ./scripts/run_ablations.sh [story_file]

set -e

STORY_FILE=${1:-"data/stories/story_01.txt"}
OUTPUT_BASE="outputs/ablations"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "=========================================="
echo "Running Ablation Studies"
echo "=========================================="
echo "Story: $STORY_FILE"
echo "Output base: $OUTPUT_BASE"
echo "=========================================="
echo ""

# Check if story exists
if [ ! -f "$STORY_FILE" ]; then
    echo "❌ Story file not found: $STORY_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Function to run experiment
run_experiment() {
    local name=$1
    local output_dir="$OUTPUT_BASE/$name"
    local config=$2

    echo ""
    echo "Running: $name"
    echo "Config: $config"
    echo "Output: $output_dir"
    echo ""

    # Create config file
    mkdir -p "$output_dir"
    echo "$config" > "$output_dir/config.txt"

    # Run pipeline with config
    python scripts/run_story_pipeline_with_agents.py \
        --storyline "$STORY_FILE" \
        --output-dir "$output_dir" \
        --max-iterations 3 \
        --quality-threshold 0.70

    # Evaluate
    python evaluation.py "$output_dir" > "$output_dir/eval.log" 2>&1

    echo "✅ $name complete"
}

# Baseline: All agents enabled
echo ""
echo ">>> BASELINE: All agents enabled"
run_experiment "baseline_all_agents" "all_agents=true"

# Ablation 1: No ContinuityAuditor
echo ""
echo ">>> ABLATION 1: Removing ContinuityAuditor"
run_experiment "ablation_no_continuity" "disable_continuity=true"

# Ablation 2: No StorybeatsChecker
echo ""
echo ">>> ABLATION 2: Removing StorybeatsChecker"
run_experiment "ablation_no_storybeats" "disable_storybeats=true"

# Ablation 3: No PhysicsValidator
echo ""
echo ">>> ABLATION 3: Removing PhysicsValidator"
run_experiment "ablation_no_physics" "disable_physics=true"

# Ablation 4: Continuity Only
echo ""
echo ">>> ABLATION 4: ContinuityAuditor only"
run_experiment "only_continuity" "only_continuity=true"

# Ablation 5: Storybeats Only
echo ""
echo ">>> ABLATION 5: StorybeatsChecker only"
run_experiment "only_storybeats" "only_storybeats=true"

# Ablation 6: Physics Only
echo ""
echo ">>> ABLATION 6: PhysicsValidator only"
run_experiment "only_physics" "only_physics=true"

# Compare results
echo ""
echo "=========================================="
echo "ABLATION RESULTS COMPARISON"
echo "=========================================="
echo ""

echo "Experiment | Avg Score | Avg Iterations | Improvement"
echo "-----------|-----------|----------------|------------"

for dir in "$OUTPUT_BASE"/*/; do
    if [ -f "$dir/convergence_report.json" ]; then
        NAME=$(basename "$dir")
        AVG_SCORE=$(grep -o '"avg_final": [0-9.]*' "$dir/convergence_report.json" | cut -d: -f2 | tr -d ' ')
        AVG_ITER=$(grep -o '"avg_iterations_per_window": [0-9.]*' "$dir/convergence_report.json" | cut -d: -f2 | tr -d ' ')
        AVG_IMPR=$(grep -o '"avg_improvement": [0-9.]*' "$dir/convergence_report.json" | cut -d: -f2 | tr -d ' ')

        printf "%-40s | %9.3f | %14.2f | %10.3f\n" "$NAME" "$AVG_SCORE" "$AVG_ITER" "$AVG_IMPR"
    fi
done

echo ""
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="
echo ""

echo "Next steps:"
echo "1. Review comparison table above"
echo "2. Check detailed reports: cat $OUTPUT_BASE/*/convergence_report.json"
echo "3. Visualize: python visualization.py $OUTPUT_BASE/baseline_all_agents"
echo ""
