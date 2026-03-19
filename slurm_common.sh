#!/bin/bash
# Shared defaults for the repository's SLURM launchers.
#
# Common variables:
# - SCENEWEAVER_USER: cluster user that owns the runtime, project, and model paths.
# - SCENEWEAVER_ACCOUNT: shared account/group prefix used in HPC and vault paths.
# - SCENEWEAVER_HPC_ROOT: writable cluster workspace for this user.
# - SCENEWEAVER_VAULT_ROOT: shared model/data storage root for this user.
# - SCENEWEAVER_PROJECT_ROOT: canonical repo path when a launcher needs an explicit default.
# - SCENEWEAVER_CONDA_SH: conda activation script used by the launchers.
# - SCENEWEAVER_DEFAULT_ENV: default conda environment used by most jobs.
# - SCENEWEAVER_PYTHON_MODULE / SCENEWEAVER_CUDA_MODULE: optional module defaults.
# - SCENEWEAVER_PROXY_URL / SCENEWEAVER_NO_PROXY: proxy defaults for compute nodes.
# - SCENEWEAVER_MODEL_STORAGE_ROOT: base path for shared local model caches.
# - SCENEWEAVER_ATUIN_ROOT: large-capacity filesystem root for oversized model caches.
# - SCENEWEAVER_WAN_MODEL_DIR: default local Wan T2V model path.
# - SCENEWEAVER_WAN_I2V_MODEL_DIR: default local Wan I2V model path.
# - SCENEWEAVER_DINOV2_MODEL_DIR: default local DINOv2 path.
# - SCENEWEAVER_FACEBOOK_DINOV2_MODEL_DIR: alternate DINOv2 path used by finetuning jobs.

SCENEWEAVER_ACCOUNT="${SCENEWEAVER_ACCOUNT:-v123be}"
SCENEWEAVER_USER="${SCENEWEAVER_USER:-v123be36}"
SCENEWEAVER_HPC_ROOT="${SCENEWEAVER_HPC_ROOT:-/home/hpc/${SCENEWEAVER_ACCOUNT}/${SCENEWEAVER_USER}}"
SCENEWEAVER_VAULT_ROOT="${SCENEWEAVER_VAULT_ROOT:-/home/vault/${SCENEWEAVER_ACCOUNT}/${SCENEWEAVER_USER}}"
SCENEWEAVER_PROJECT_ROOT="${SCENEWEAVER_PROJECT_ROOT:-${SCENEWEAVER_HPC_ROOT}/Multimodal_Final_SceneWeaver}"

SCENEWEAVER_CONDA_SH="${SCENEWEAVER_CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
SCENEWEAVER_DEFAULT_ENV="${SCENEWEAVER_DEFAULT_ENV:-sceneweaver_runtime}"
SCENEWEAVER_PYTHON_MODULE="${SCENEWEAVER_PYTHON_MODULE:-python/3.12-conda}"
SCENEWEAVER_CUDA_MODULE="${SCENEWEAVER_CUDA_MODULE:-cuda/12.4.1}"

SCENEWEAVER_PROXY_URL="${SCENEWEAVER_PROXY_URL:-http://proxy:80}"
SCENEWEAVER_NO_PROXY="${SCENEWEAVER_NO_PROXY:-localhost,127.0.0.1,::1}"

SCENEWEAVER_MODEL_STORAGE_ROOT="${SCENEWEAVER_MODEL_STORAGE_ROOT:-${SCENEWEAVER_VAULT_ROOT}}"
SCENEWEAVER_ATUIN_ROOT="${SCENEWEAVER_ATUIN_ROOT:-/home/atuin/${SCENEWEAVER_ACCOUNT}/${SCENEWEAVER_USER}}"
SCENEWEAVER_WAN_MODEL_DIR="${SCENEWEAVER_WAN_MODEL_DIR:-${SCENEWEAVER_MODEL_STORAGE_ROOT}/Wan2.1-T2V-1.3B-Diffusers}"
SCENEWEAVER_WAN_I2V_MODEL_DIR="${SCENEWEAVER_WAN_I2V_MODEL_DIR:-${SCENEWEAVER_ATUIN_ROOT}/Wan2.1-I2V-14B-480P-Diffusers}"
SCENEWEAVER_DINOV2_MODEL_DIR="${SCENEWEAVER_DINOV2_MODEL_DIR:-${SCENEWEAVER_MODEL_STORAGE_ROOT}/dinov2-base}"
SCENEWEAVER_FACEBOOK_DINOV2_MODEL_DIR="${SCENEWEAVER_FACEBOOK_DINOV2_MODEL_DIR:-${SCENEWEAVER_MODEL_STORAGE_ROOT}/facebook/dinov2-base}"

sceneweaver_print_vars() {
  local key

  for key in "$@"; do
    printf '%s=%s\n' "${key}" "${!key:-}"
  done
}
