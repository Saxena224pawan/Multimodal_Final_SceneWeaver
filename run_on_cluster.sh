#!/bin/bash
#SBATCH --job-name=sceneweaver_multiagent
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/multiagent_%j.log
#SBATCH --error=logs/multiagent_%j.err

# Load your environment
module load cuda
module load python/3.11  # or your Python version

# Activate your conda environment if you have one
# conda activate your_env

# Navigate to project
cd ~/workspace_mac/Multimodal_Final_SceneWeaver

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "SceneWeaver Multi-Agent Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="
echo ""

# Run data collection on 10 stories
echo "Starting data collection..."
bash scripts/collect_data.sh 10 3

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Data collection complete!"
    echo ""

    # Run ablation studies
    echo "Starting ablation studies..."
    bash scripts/run_ablations.sh data/stories/story_01.txt

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Ablation studies complete!"
        echo ""

        # Aggregate results
        echo "Aggregating results..."
        python3 scripts/aggregate_results.py outputs/multi_agent_full_run

        echo ""
        echo "=========================================="
        echo "✅ ALL TASKS COMPLETE!"
        echo "=========================================="
        echo "Results saved to: outputs/multi_agent_full_run/"
        echo "Summary: outputs/multi_agent_full_run/AGGREGATE_RESULTS.json"
        echo "=========================================="
    else
        echo "❌ Ablation studies failed"
        exit 1
    fi
else
    echo "❌ Data collection failed"
    exit 1
fi
