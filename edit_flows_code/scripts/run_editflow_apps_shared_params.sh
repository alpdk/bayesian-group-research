#!/usr/bin/env bash
# APPS, prompt/target 512+512 bytes, global batch 64, 20k steps, hidden-test eval.
set -euo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${EDITFLOW_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
UNI_D2="$(cd "$ROOT/.." && pwd)"
PY="${PYTHON:-${UNI_D2}/.venv/bin/python}"
export PYTHONPATH="${UNI_D2}/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-edit-diffusion}"
cd "$ROOT"
SAVE_DIR="${SAVE_DIR:-$ROOT/results/editflow-apps-L512-shared-params}"
mkdir -p "$SAVE_DIR"
"$PY" train.py \
  --dataset apps \
  --steps 20000 \
  --batch-size 64 \
  --grad-accum 1 \
  --lr 3e-4 \
  --weight-decay 0.03 \
  --warmup-steps 500 \
  --hidden-dim 768 \
  --num-layers 12 \
  --num-heads 12 \
  --dropout 0.1 \
  --max-prompt-len 512 \
  --max-target-len 512 \
  --coupling empty \
  --val-every 2000 \
  --patience 5 \
  --grad-clip 1.0 \
  --t-eps 1e-2 \
  --precision bf16 \
  --task-eval-samples 32 \
  --task-eval-steps 128 \
  --val-max-examples 256 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name editflow-apps-L512-shared-params \
  --save-dir "$SAVE_DIR" \
  --save-every 2000 \
  --log-every 50 \
  --seed 42 \
  "$@"
