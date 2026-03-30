#!/bin/bash -l
#SBATCH --job-name=ft_pororo_cont
#SBATCH --output=slurm_logs/ft_pororo_cont_%j.out
#SBATCH --error=slurm_logs/ft_pororo_cont_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver}"
DATASET_ROOT="${DATASET_ROOT:-/home/vault/v123be/v123be36/PororoSV}"
DINO_MODEL_ID="${DINO_MODEL_ID:-/home/vault/v123be/v123be36/facebook/dinov2-base}"
CONTINUITY_ADAPTER_PATH="${CONTINUITY_ADAPTER_PATH:-${PROJECT_ROOT}/outputs/pororo_continuity_adapter.pt}"

DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEMPERATURE="${TEMPERATURE:-0.07}"
LR_PROJECTOR="${LR_PROJECTOR:-5e-5}"
LR_BACKBONE="${LR_BACKBONE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
VAL_SPLIT="${VAL_SPLIT:-unseen}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
PROJ_DIM="${PROJ_DIM:-512}"
UNFREEZE_BACKBONE="${UNFREEZE_BACKBONE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate sceneweaver_runtime

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "DINO_MODEL_ID=${DINO_MODEL_ID}"
echo "CONTINUITY_ADAPTER_PATH=${CONTINUITY_ADAPTER_PATH}"
echo "DEVICE=${DEVICE}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "VAL_SPLIT=${VAL_SPLIT}"
echo "MAX_TRAIN_PAIRS=${MAX_TRAIN_PAIRS}"
echo "MAX_VAL_PAIRS=${MAX_VAL_PAIRS}"
echo "UNFREEZE_BACKBONE=${UNFREEZE_BACKBONE}"
echo "NUM_WORKERS=${NUM_WORKERS}"

CMD=(python scripts/finetune_pororo_continuity.py
  --dataset_root "${DATASET_ROOT}"
  --dino_model_id "${DINO_MODEL_ID}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  --lr_projector "${LR_PROJECTOR}"
  --lr_backbone "${LR_BACKBONE}"
  --weight_decay "${WEIGHT_DECAY}"
  --hidden_dim "${HIDDEN_DIM}"
  --proj_dim "${PROJ_DIM}"
  --num_workers "${NUM_WORKERS}"
  --val_split "${VAL_SPLIT}"
  --max_train_pairs "${MAX_TRAIN_PAIRS}"
  --max_val_pairs "${MAX_VAL_PAIRS}"
  --save_path "${CONTINUITY_ADAPTER_PATH}")

if [ "${UNFREEZE_BACKBONE}" = "1" ]; then
  CMD+=(--unfreeze_backbone)
fi

"${CMD[@]}"
