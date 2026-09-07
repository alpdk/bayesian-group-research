#!/usr/bin/env bash
# TinyGSM 100k, prompt/target 128+128, global batch 64, 260k steps, GSM8K eval.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
WS="${WORKSPACE:-/data/horse/ws/alpo658h-insert-diff}"
ROOT="${EDITFLOW_ROOT:-$WS/projects/edit_flows_code}"
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
SAVE_DIR="${SAVE_DIR:-$WS/scratch/editflow-tinygsm-L256-shared-params}"
mkdir -p "$SAVE_DIR"
echo "RUN_MARKER host=$(hostname) iso=$(date --iso-8601=seconds) cuda=$($PY -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "RUN_MARKER train_start iso=$(date --iso-8601=seconds)"
RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  RESUME_ARGS+=(--resume "$RESUME")
fi
"$PY" train.py \
  --dataset tinygsm \
  --max-examples 100000 \
  --steps 260000 \
  --batch-size 64 \
  --grad-accum 1 \
  --lr 3e-4 \
  --weight-decay 0.03 \
  --warmup-steps 500 \
  --hidden-dim 768 \
  --num-layers 12 \
  --num-heads 12 \
  --dropout 0.1 \
  --max-prompt-len 128 \
  --max-target-len 128 \
  --coupling empty \
  --val-every 2000 \
  --patience 9999 \
  --grad-clip 1.0 \
  --t-eps 1e-2 \
  --precision bf16 \
  --task-eval-samples 32 \
  --task-eval-steps 128 \
  --val-max-examples 256 \
  --wandb-project edit-diffusion \
  --wandb-name editflow-tinygsm-L256-shared-params \
  --save-dir "$SAVE_DIR" \
  --save-every 2000 \
  --log-every 50 \
  --seed 42 \
  "${RESUME_ARGS[@]}"
echo "RUN_MARKER train_end iso=$(date --iso-8601=seconds) status=$?"
BEST="$SAVE_DIR/model_best.pt"
if [ ! -f "$BEST" ]; then
  BEST="$SAVE_DIR/model.pt"
fi
if [ -f "$BEST" ]; then
  echo "RUN_MARKER eval_start iso=$(date --iso-8601=seconds) ckpt=$BEST"
  "$PY" sample.py --checkpoint "$BEST" --num-prompts 0 --num-steps 128 --batch-size 8 \
    --max-gen-len 128 --out "$SAVE_DIR/gsm8k_test.json" \
    --wandb-project edit-diffusion \
    --wandb-name editflow-tinygsm-L256-shared-params-eval
  echo "RUN_MARKER eval_end iso=$(date --iso-8601=seconds) status=$?"
fi
