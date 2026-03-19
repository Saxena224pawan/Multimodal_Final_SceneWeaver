#!/bin/bash -l
#SBATCH --job-name=ft_pororo_cont
#SBATCH --output=slurm_logs/ft_pororo_cont_%j.out
#SBATCH --error=slurm_logs/ft_pororo_cont_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

set -euo pipefail

COMMON_SLURM_ROOT="${COMMON_SLURM_ROOT:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}}"
COMMON_SLURM_SH="${COMMON_SLURM_ROOT}/slurm_common.sh"
# shellcheck source=./slurm_common.sh
source "${COMMON_SLURM_SH}"

PROJECT_ROOT="${PROJECT_ROOT:-${SCENEWEAVER_PROJECT_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${SCENEWEAVER_VAULT_ROOT}/PororoSV}"
DINO_MODEL_ID="${DINO_MODEL_ID:-${SCENEWEAVER_FACEBOOK_DINOV2_MODEL_DIR}}"
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

CONDA_SH="${CONDA_SH:-${SCENEWEAVER_CONDA_SH}}"
ENV_PATH="${ENV_PATH:-${SCENEWEAVER_DEFAULT_ENV}}"
source "${CONDA_SH}"
conda activate "${ENV_PATH}"

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
