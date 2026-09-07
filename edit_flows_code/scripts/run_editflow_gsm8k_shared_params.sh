#!/usr/bin/env bash
# GSM8K EditFlow shared-params. Knobs: configs/editflow/gsm8k.yaml
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
WS="${WORKSPACE:-/data/horse/ws/alpo658h-insert-diff}"
EDITFLOW_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPARISON_ROOT="${COMPARISON_ROOT:-$(cd "$EDITFLOW_DIR/.." && pwd)}"
ROOT="${EDITFLOW_ROOT:-$EDITFLOW_DIR}"
CFG="${CFG:-$COMPARISON_ROOT/configs/editflow/gsm8k.yaml}"
PY="${PYTHON:-$WS/conda_envs/uni-d2/bin/python}"
export PATH="$(dirname "$PY"):${PATH}"
export HF_HOME="${HF_HOME:-$WS/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WS/.cache}"
export WANDB_PROJECT=edit-diffusion
export WANDB_MODE=online
export WANDB_DIR="$ROOT/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
cd "$ROOT"
SAVE_DIR="${SAVE_DIR:-$WS/scratch/editflow-gsm8k-L256-shared-params}"
mkdir -p "$SAVE_DIR"
echo "RUN_MARKER host=$(hostname) iso=$(date --iso-8601=seconds) cuda=$($PY -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "RUN_MARKER train_start iso=$(date --iso-8601=seconds) cfg=$CFG"
"$PY" train.py --cfg "$CFG" --save-dir "$SAVE_DIR"
echo "RUN_MARKER train_end iso=$(date --iso-8601=seconds) status=$?"
