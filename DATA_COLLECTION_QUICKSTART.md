# 🚀 Days 4-5 Ready: Data Collection Automation

## Quick Start (30 seconds)

### Test Everything Works (Mock Models)
```bash
# Runs on 2 mock windows, finishes in ~5 seconds
python test_integration.py
```

✅ If this works, you're ready for real models!

---

## Real Model Testing (30 min - hours depending on models)

### Step 1: Verify Models Are Available

```bash
# Check what models you have
ls -la video_backbone/models/  # WAN I2V
ls -la director_llm/models/    # LLM
ls -la memory_module/models/   # Embedder
```

### Step 2: Test on 1 Story (Diagnostic Run)

```bash
# Single story, 2 windows only
python scripts/run_story_pipeline_with_agents.py \
    --storyline data/stories/story_01.txt \
    --output-dir outputs/diagnostic_run \
    --num-test-stories 2 \
    --max-iterations 3

# Check if it worked
cat outputs/diagnostic_run/summary.json
```

**Output should show**:
- ✅ 2 windows processed
- ✅ Convergence report generated
- ✅ Metadata files saved

---

## Full Data Collection (Days 4-5)

### Option A: Sequential (Safer, More Reproducible)

```bash
# Make scripts executable
chmod +x scripts/collect_data.sh
chmod +x scripts/run_ablations.sh
chmod +x scripts/aggregate_results.py

# Run on 10 stories sequentially
./scripts/collect_data.sh 10 3

# This will:
# - Generate outputs/multi_agent_full_run/run_001, run_002, ... run_010
# - Evaluate each one automatically
# - Show convergence stats for each
# - Take ~2-4 hours depending on model speed
```

### Option B: Quick Test First

```bash
# Run on 3 stories just to validate setup
./scripts/collect_data.sh 3 2

# Should complete in ~30 min - 1 hour
# Then run full collection:
./scripts/collect_data.sh 10 3
```

---

## Ablation Studies (Measure Agent Importance)

### Run All Ablations on 1 Story

```bash
# Test all agent configurations on story_01
./scripts/run_ablations.sh data/stories/story_01.txt

# Outputs to: outputs/ablations/
# Creates 6 runs:
#   - baseline_all_agents
#   - ablation_no_continuity
#   - ablation_no_storybeats
#   - ablation_no_physics
#   - only_continuity
#   - only_storybeats
#   - only_physics

# Will show comparison table when done
```

---

## Aggregate & Analyze Results

```bash
# Combine all results into summary
python scripts/aggregate_results.py outputs/multi_agent_full_run

# Creates: outputs/multi_agent_full_run/AGGREGATE_RESULTS.json

# View summary
cat outputs/multi_agent_full_run/AGGREGATE_RESULTS.json | head -50
```

---

## Create Visualizations

```bash
# Generate publication-quality plots
python visualization.py outputs/multi_agent_full_run/run_001

# Creates outputs/multi_agent_full_run/run_001/plots/
# Files:
#   - convergence.png
#   - agent_importance.png
#   - iteration_distribution.png
#   - score_improvement.png

# Collect all plots
mkdir -p outputs/final_plots
cp outputs/multi_agent_full_run/run_*/plots/*.png outputs/final_plots/
```

---

## Complete Workflow (Days 4-5)

```bash
# --- DAY 4 MORNING (30 min - 1 hour) ---
# 1. Test setup with diagnostic run
python scripts/run_story_pipeline_with_agents.py \
    --storyline data/stories/story_01.txt \
    --output-dir outputs/diagnostic_run \
    --num-test-stories 2

# Verify it worked
python evaluation.py outputs/diagnostic_run

# --- DAY 4 AFTERNOON/EVENING (2-4 hours) ---
# 2. Full data collection on 10 stories
./scripts/collect_data.sh 10 3

# Monitor progress - should see output like:
# ✅ Story 1 complete - Windows: 5, Avg Iter: 1.2, Score: 0.76
# ✅ Story 2 complete - Windows: 5, Avg Iter: 1.1, Score: 0.79
# ... etc

# --- DAY 5 MORNING (1-2 hours) ---
# 3. Run ablation studies on subset
./scripts/run_ablations.sh data/stories/story_01.txt

# View ablation comparison table that's printed

# --- DAY 5 AFTERNOON (30 min) ---
# 4. Aggregate all results
python scripts/aggregate_results.py outputs/multi_agent_full_run

# 5. Generate visualizations
for i in {1..10}; do
    python visualization.py outputs/multi_agent_full_run/run_$(printf "%03d" $i)
done

# 6. Collect final plots
mkdir -p outputs/final_plots
cp outputs/multi_agent_full_run/run_*/plots/*.png outputs/final_plots/
```

---

## Expected Output Structure

```
outputs/
├── diagnostic_run/
│   ├── metadata/
│   ├── summary.json
│   ├── convergence_report.json
│   └── ablation_report.json
│
├── multi_agent_full_run/
│   ├── run_001/
│   │   ├── metadata/ (5-10 windows)
│   │   ├── convergence_report.json
│   │   ├── ablation_report.json
│   │   └── plots/
│   │       ├── convergence.png
│   │       ├── agent_importance.png
│   │       └── ...
│   ├── run_002/
│   ├── ... run_010/
│   └── AGGREGATE_RESULTS.json
│
├── ablations/
│   ├── baseline_all_agents/
│   ├── ablation_no_continuity/
│   ├── ablation_no_storybeats/
│   ├── ablation_no_physics/
│   ├── only_continuity/
│   ├── only_storybeats/
│   └── only_physics/
│
└── final_plots/
    ├── convergence_all.png
    ├── agent_importance_all.png
    └── ...
```

---

## Expected Results (from Mock Data)

After full collection:
- **50-100 video windows processed**
- **Convergence stats**: 1.1-1.5 avg iterations
- **Quality scores**: 0.70-0.85 average
- **Threshold hit rate**: 80-95% on first pass
- **Agent importance**: All agents contribute measurably
- **Runtime**: 2-4 hours for 10 stories (model dependent)

---

## Troubleshooting

### Models Load But Generation Fails
```bash
# Check CUDA memory
nvidia-smi

# Try reducing batch size:
# Edit scripts/run_story_pipeline_with_agents.py
# Change: num_frames=49 → num_frames=24
# Change: height=720 → height=480
```

### Script Permissions Issue
```bash
chmod +x scripts/collect_data.sh
chmod +x scripts/run_ablations.sh
python scripts/aggregate_results.py  # Always use python3 for this
```

### Models Not Found
```bash
# Check your actual model paths
export VIDEO_MODEL="/path/to/wan"
export DIRECTOR_MODEL="/path/to/qwen"

# Update scripts/run_story_pipeline_with_agents.py:
# Change: VIDEO_MODEL_PATH = "..." to your actual path
```

---

## Key Metrics to Track

After data collection, you'll have:

```json
{
  "total_windows": 50,
  "avg_iterations": 1.3,
  "single_pass_success": 0.82,
  "avg_quality_score": 0.755,
  "agent_contributions": {
    "continuity": 0.35,
    "storybeats": 0.40,
    "physics": 0.25
  }
}
```

These numbers become your **paper results**! 📊

---

## Days 6-7: Ready for Paper

Once data collection is complete, we'll:
1. Create publication-quality figures with real data ✅
2. Add statistical significance tests
3. Write the research paper
4. Create final GitHub release

---

## Command Reference

```bash
# Test
python test_integration.py

# Diagnose
python scripts/run_story_pipeline_with_agents.py \
    --storyline data/stories/story_01.txt \
    --output-dir outputs/diagnostic_run \
    --num-test-stories 2

# Collect
./scripts/collect_data.sh [num_stories] [max_iterations]

# Ablate
./scripts/run_ablations.sh data/stories/story_01.txt

# Aggregate
python scripts/aggregate_results.py outputs/multi_agent_full_run

# Visualize
python visualization.py outputs/multi_agent_full_run/run_001

# Evaluate
python evaluation.py outputs/multi_agent_full_run/run_001
```

---

## Status: READY FOR EXECUTION 🚀

All scripts are prepared and documented. Ready to collect real data and get publication-ready results!

**Estimated time remaining**: 2-3 days (4-5 for full paper)

Start with: `./scripts/collect_data.sh 3 2` for quick validation
