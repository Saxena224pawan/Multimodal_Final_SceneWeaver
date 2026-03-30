#!/bin/bash
# Data Collection Script - Automated Multi-Agent System Testing
# Usage: ./scripts/collect_data.sh [num_stories] [max_iterations]

set -e

PYTHON_BIN=${PYTHON_BIN:-python}

# Configuration
NUM_STORIES=${1:-10}
MAX_ITERATIONS=${2:-3}
QUALITY_THRESHOLD=0.70
OUTPUT_BASE="outputs/multi_agent_full_run"
DATA_DIR="data/stories"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "=========================================="
echo "Multi-Agent Data Collection"
echo "=========================================="
echo "Stories: $NUM_STORIES"
echo "Max iterations: $MAX_ITERATIONS"
echo "Output base: $OUTPUT_BASE"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"
RESULTS_FILE="$OUTPUT_BASE/collection_results_$TIMESTAMP.json"

# Initialize results JSON
cat > "$RESULTS_FILE" << 'EOF'
{
  "timestamp": "placeholder",
  "total_stories": 0,
  "total_windows": 0,
  "completed_runs": []
}
EOF

# Counter
COMPLETED=0
FAILED=0
TOTAL_WINDOWS=0

# Run on each story
for i in $(seq 1 $NUM_STORIES); do
    STORY_FILE="$DATA_DIR/story_$(printf "%02d" $i).txt"
    RUN_DIR="$OUTPUT_BASE/run_$(printf "%03d" $i)"

    # Check if story exists
    if [ ! -f "$STORY_FILE" ]; then
        echo "⚠️  Story $i not found: $STORY_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Story $i / $NUM_STORIES"
    echo "=========================================="
    echo "File: $STORY_FILE"
    echo "Output: $RUN_DIR"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Run pipeline
    if "$PYTHON_BIN" scripts/run_story_pipeline_with_agents.py \
        --storyline "$STORY_FILE" \
        --output-dir "$RUN_DIR" \
        --max-iterations "$MAX_ITERATIONS" \
        --quality-threshold "$QUALITY_THRESHOLD"; then

        # Evaluate run
        echo ""
        echo "Evaluating run..."
        "$PYTHON_BIN" evaluation.py "$RUN_DIR" > "$RUN_DIR/evaluation_output.log" 2>&1

        # Extract results
        if [ -f "$RUN_DIR/convergence_report.json" ]; then
            WINDOWS=$(grep -o '"total_windows": [0-9]*' "$RUN_DIR/convergence_report.json" | cut -d: -f2 | tr -d ' ')
            AVG_ITER=$(grep -o '"avg_iterations_per_window": [0-9.]*' "$RUN_DIR/convergence_report.json" | cut -d: -f2 | tr -d ' ')
            AVG_SCORE=$(grep -o '"avg_final": [0-9.]*' "$RUN_DIR/convergence_report.json" | cut -d: -f2 | tr -d ' ')

            echo "✅ Story $i complete"
            echo "   Windows: $WINDOWS"
            echo "   Avg iterations: $AVG_ITER"
            echo "   Avg score: $AVG_SCORE"

            COMPLETED=$((COMPLETED + 1))
            TOTAL_WINDOWS=$((TOTAL_WINDOWS + WINDOWS))
        fi

        sleep 2  # Brief pause between runs

    else
        echo "❌ Story $i failed"
        FAILED=$((FAILED + 1))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Completed: $COMPLETED / $NUM_STORIES"
echo "Failed: $FAILED / $NUM_STORIES"
echo "Total windows: $TOTAL_WINDOWS"
echo "Results file: $RESULTS_FILE"
echo "=========================================="
echo ""

if [ $COMPLETED -gt 0 ]; then
    echo "✅ Data collection successful!"
    echo ""
    echo "Next steps:"
    echo "1. Aggregate results: python scripts/aggregate_results.py"
    echo "2. Visualize: python visualization.py $OUTPUT_BASE/run_001"
    echo "3. View summary: cat $OUTPUT_BASE/run_001/convergence_report.json"
    echo ""
else
    echo "❌ No successful runs. Check logs above."
    exit 1
fi
