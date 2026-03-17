# Days 4-5: Real Data Collection Guide

## Overview
This guide walks you through collecting experimental data from the multi-agent system running on real models and real stories. Expected timeline: **2-3 days**.

---

## Step 1: Verify Model Availability

Check which models you have access to:

```bash
# Check existing models in pipeline
grep -r "model_id\|MODEL" scripts/run_story_pipeline.py | head -20

# List available model directories
ls -la /path/to/models/

# Check your existing config
cat configs/default.yaml
```

### Models You'll Need

| Model | Purpose | Where | Optional? |
|-------|---------|-------|-----------|
| **Wan I2V** | Video generation | `video_backbone/` | ❌ Required |
| **Qwen/LLM** | Story planning + agents | `director_llm/` | ❌ Required |
| **CLIP/DINOv2** | Embeddings | `memory_module/` | ❌ For ContinuityAuditor |
| **BLIP-2/LLaVA** | Captioning | `Caption_Gen/` | ❌ For captions |

---

## Step 2: Connect Real Models

### Option A: Use Existing Pipeline Config

If models are already integrated:

```bash
# Use existing wrapper script
python scripts/run_story_pipeline.py \
    --storyline "My story" \
    --video-model-id <model_path> \
    --director-model-id <llm_path> \
    --output-dir outputs/baseline_run
```

### Option B: Adapt run_story_pipeline_with_agents.py

Edit the script to use your specific model paths:

```python
# In scripts/run_story_pipeline_with_agents.py, update:

def main() -> None:
    args = parse_args()

    # ✏️ CUSTOMIZE THESE PATHS
    VIDEO_MODEL_PATH = "/path/to/wan_i2v_model"
    DIRECTOR_MODEL_PATH = "/path/to/qwen_model"
    EMBEDDER_MODEL = "clip"  # or "dinov2"

    # Load with custom paths
    video_config = WanBackboneConfig(model_id=VIDEO_MODEL_PATH)
    director_config = SceneDirectorConfig(model_id=DIRECTOR_MODEL_PATH)
    # ... rest of code
```

### Option C: Use Mock Models with Placeholder Timing

For quick testing (generates results instantly):

```bash
# Uses mock models from test_integration.py
python test_integration.py  # Already working!
```

---

## Step 3: Prepare Test Stories

Create 10-20 diverse stories:

```bash
mkdir -p data/stories

# Story 1: Short action
cat > data/stories/story_01.txt << 'EOF'
A hunter in the forest discovers ancient ruins while tracking prey.
He investigates the carved symbols and finds a hidden chamber filled
with mysterious artifacts that glow with an unknown energy.
EOF

# Story 2: Dialogue-heavy
cat > data/stories/story_02.txt << 'EOF'
Two travelers meet at a crossroads inn. They share stories over dinner,
revealing their mutual past. As distrust turns to understanding, they
decide to journey together toward the mountains.
EOF

# Story 3: Emotional journey
cat > data/stories/story_03.txt << 'EOF'
A young child searches for their lost pet in the neighborhood.
After hours of searching, they find the pet in a hidden garden.
The reunion brings tears of joy and a new friendship with the garden's keeper.
EOF

# ... create 10-20 total stories
```

---

## Step 4: Run Data Collection (Days 4-5)

### Quick Start (Small Test First)

```bash
# Test on 2 stories first
python scripts/run_story_pipeline_with_agents.py \
    --storyline data/stories/story_01.txt \
    --output-dir outputs/run_001 \
    --max-iterations 3 \
    --quality-threshold 0.70 \
    --num-test-stories 2  # Only 2 windows to test
```

Check results:
```bash
python evaluation.py outputs/run_001
# Should show convergence stats
```

### Full Data Collection (10-20 Stories)

**Option 1: Sequential (Recommended for first run)**

```bash
#!/bin/bash
# save as: scripts/collect_data.sh

OUTPUT_BASE="outputs/multi_agent_full_run"
DATA_DIR="data/stories"

mkdir -p "$OUTPUT_BASE"

# Run on each story sequentially
for i in {1..10}; do
    STORY_FILE="$DATA_DIR/story_$(printf "%02d" $i).txt"

    if [ ! -f "$STORY_FILE" ]; then
        echo "⚠️  $STORY_FILE not found, skipping..."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Processing Story $i / 10"
    echo "=========================================="

    python scripts/run_story_pipeline_with_agents.py \
        --storyline "$STORY_FILE" \
        --output-dir "$OUTPUT_BASE/run_$(printf "%03d" $i)" \
        --max-iterations 3 \
        --quality-threshold 0.70

    echo "✅ Story $i complete"
    sleep 5  # Brief pause between runs
done

echo ""
echo "✅ All stories processed!"
```

Run it:
```bash
chmod +x scripts/collect_data.sh
./scripts/collect_data.sh
```

**Option 2: Parallel (Faster, if you have multiple GPUs)**

```bash
#!/bin/bash
# Run 2 stories in parallel
for i in {1..10..2}; do
    python scripts/run_story_pipeline_with_agents.py \
        --storyline "data/stories/story_$(printf "%02d" $i).txt" \
        --output-dir "outputs/multi_agent_full_run/run_$(printf "%03d" $i)" &

    python scripts/run_story_pipeline_with_agents.py \
        --storyline "data/stories/story_$(printf "%02d" $((i+1))).txt" \
        --output-dir "outputs/multi_agent_full_run/run_$(printf "%03d" $((i+1)))" &

    wait  # Wait for both to finish before next iteration
done
```

---

## Step 5: Collect Convergence Data

After running all stories:

```bash
# Aggregate all results
python evaluation.py outputs/multi_agent_full_run/run_001
python evaluation.py outputs/multi_agent_full_run/run_002
# ... repeat for all runs

# Or create aggregation script:
for run_dir in outputs/multi_agent_full_run/run_*; do
    python evaluation.py "$run_dir"
done
```

---

## Step 6: Run Ablation Studies

### Ablation 1: Remove ContinuityAuditor

```python
# Create: scripts/ablation_no_continuity.py

from multi_agent_refinement.agents import StorybeatsChecker, PhysicsValidator
from multi_agent_refinement.refinement_engine import RefinementEngine

# Use engine without ContinuityAuditor
engine = RefinementEngine(
    video_model=video_model,
    captioner=captioner,
    embedding_model=embedder,
    llm_model=llm_model,
    config={
        "max_iterations": 3,
        "quality_threshold": 0.70,
    }
)

# Modify engine to skip continuity agent
# Run same stories with modified weights
```

### Ablation 2: Remove StorybeatsChecker

```python
# Similar pattern - set weight to 0 or remove agent
```

### Ablation 3: Remove PhysicsValidator

```python
# Similar pattern - set weight to 0 or remove agent
```

### Automated Ablation Runner

```bash
#!/bin/bash
# save as: scripts/run_ablations.sh

BASE_STORY="data/stories/story_01.txt"
BASE_OUTPUT="outputs/ablations"

mkdir -p "$BASE_OUTPUT"

echo "Running ablation studies..."
echo ""

# Baseline (all agents)
echo "Baseline (all agents)..."
python scripts/run_story_pipeline_with_agents.py \
    --storyline "$BASE_STORY" \
    --output-dir "$BASE_OUTPUT/baseline" \
    --max-iterations 3

# Ablation 1: No continuity
echo "Ablation 1: No continuity..."
DISABLE_CONTINUITY=1 python scripts/run_story_pipeline_with_agents.py \
    --storyline "$BASE_STORY" \
    --output-dir "$BASE_OUTPUT/no_continuity" \
    --max-iterations 3

# Ablation 2: No storybeats
echo "Ablation 2: No storybeats..."
DISABLE_STORYBEATS=1 python scripts/run_story_pipeline_with_agents.py \
    --storyline "$BASE_STORY" \
    --output-dir "$BASE_OUTPUT/no_storybeats" \
    --max-iterations 3

# Ablation 3: No physics
echo "Ablation 3: No physics..."
DISABLE_PHYSICS=1 python scripts/run_story_pipeline_with_agents.py \
    --storyline "$BASE_STORY" \
    --output-dir "$BASE_OUTPUT/no_physics" \
    --max-iterations 3

echo ""
echo "✅ Ablation studies complete!"

# Generate comparison
python evaluation.py "$BASE_OUTPUT/baseline"
python evaluation.py "$BASE_OUTPUT/no_continuity"
python evaluation.py "$BASE_OUTPUT/no_storybeats"
python evaluation.py "$BASE_OUTPUT/no_physics"
```

---

## Step 7: Aggregate Results

Create comprehensive summary:

```python
# save as: scripts/aggregate_results.py

import json
from pathlib import Path

results = {}

# Load all convergence reports
for run_dir in sorted(Path("outputs/multi_agent_full_run").glob("run_*")):
    report_file = run_dir / "convergence_report.json"
    if report_file.exists():
        with open(report_file) as f:
            results[run_dir.name] = json.load(f)

# Compute aggregate statistics
from evaluation import compute_convergence_stats, load_metadata

all_metadatas = []
for run_dir in sorted(Path("outputs/multi_agent_full_run").glob("run_*")):
    metadata_dir = run_dir / "metadata"
    if metadata_dir.exists():
        all_metadatas.extend(load_metadata(metadata_dir))

aggregate_stats = compute_convergence_stats(all_metadatas)

# Save aggregate
aggregate = {
    "total_runs": len(results),
    "total_windows": len(all_metadatas),
    "average_iterations": aggregate_stats.avg_iterations,
    "threshold_hit_rate": (
        aggregate_stats.windows_at_threshold_1 / aggregate_stats.total_windows
    ) if aggregate_stats.total_windows > 0 else 0,
    "avg_score_improvement": aggregate_stats.score_improvement,
    "per_run_results": results,
}

with open("outputs/AGGREGATE_RESULTS.json", "w") as f:
    json.dump(aggregate, f, indent=2)

print(json.dumps(aggregate, indent=2))
```

Run it:
```bash
python scripts/aggregate_results.py
```

---

## Step 8: Generate Visualizations

```bash
# Visualize all runs
for run_dir in outputs/multi_agent_full_run/run_*; do
    python visualization.py "$run_dir"
done

# Collect all plots
mkdir -p outputs/all_plots
cp outputs/multi_agent_full_run/run_*/plots/*.png outputs/all_plots/

# Create composite report
convert outputs/all_plots/*.png outputs/all_plots/composite.pdf
```

---

## Timeline Estimate (Days 4-5)

| Task | Duration | Status |
|------|----------|--------|
| Model setup | 30 min | ⏳ |
| Story preparation | 1 hour | ⏳ |
| Small test run (2 stories) | 30 min - 2 hours* | ⏳ |
| Full data collection (10 stories) | 4-8 hours* | ⏳ |
| Ablation studies | 2-4 hours* | ⏳ |
| Result aggregation | 30 min | ⏳ |
| Visualization | 30 min | ⏳ |

*Depends on model inference speed

---

## Expected Results

### From Convergence Analysis
```
Total Windows: 50 (5 stories × 10 windows each)
Avg Iterations: 1.5-2.0  (> 70% on first pass)
Avg Score: 0.72-0.78
Threshold Hit Rate: 85-95%
Quality Improvement per Iteration: 0.05-0.10
```

### From Ablation Studies
Expected agent importance:
- **ContinuityAuditor**: High (embeddings very useful)
- **StorybeatsChecker**: High (narrative critical)
- **PhysicsValidator**: Medium (nice to have)

---

## Troubleshooting

### Problem: Models fail to load
```bash
# Check available models
ls -la /path/to/model/

# Try loading individually
python -c "from video_backbone import WanBackbone; print('✓')"
python -c "from director_llm import SceneDirector; print('✓')"
```

### Problem: Memory errors
```bash
# Reduce batch size / number of frames
# Edit scripts/run_story_pipeline_with_agents.py
num_frames=24  # instead of 49
height=480  # instead of 720
```

### Problem: Slow inference
```bash
# Enable optimizations
export CUDA_VISIBLE_DEVICES=0  # Single GPU
python scripts/run_story_pipeline_with_agents.py ... --num-test-stories 1
```

---

## Quick Checklist (Before Starting)

- [ ] Models loaded and tested
- [ ] 10-20 stories prepared in `data/stories/`
- [ ] `run_story_pipeline_with_agents.py` configured with correct model paths
- [ ] Sufficient disk space (~5-10GB for 10 stories × 8 sec each)
- [ ] GPU memory available (test with 1 window)
- [ ] Output directories configured
- [ ] Evaluation & visualization modules working

---

## Commands Summary

```bash
# 1. Quick test
python scripts/run_story_pipeline_with_agents.py \
    --storyline data/stories/story_01.txt \
    --output-dir outputs/test_real_models \
    --num-test-stories 2

# 2. Evaluate
python evaluation.py outputs/test_real_models

# 3. Run full collection
./scripts/collect_data.sh

# 4. Run ablations
./scripts/run_ablations.sh

# 5. Aggregate
python scripts/aggregate_results.py

# 6. Visualize
python visualization.py outputs/multi_agent_full_run/run_001

# 7. View results
cat outputs/AGGREGATE_RESULTS.json
```

---

## Next: Days 6-7 (Paper Writing)

Once you have real results in `outputs/AGGREGATE_RESULTS.json`, we'll:
1. Create publication-ready figures with real data
2. Add statistical significance tests
3. Write the paper with actual numbers
4. Create final release

Ready? Start with **Step 1** above! 🚀
