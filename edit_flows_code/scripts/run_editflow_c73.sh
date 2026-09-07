#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
WS=/data/horse/ws/alpo658h-insert-diff
ROOT="$WS/projects/edit_flows_code"
PY="$WS/conda_envs/uni-d2/bin/python"
export PATH="$WS/conda_envs/uni-d2/bin:$PATH"
export HF_HOME="${HF_HOME:-$WS/hf_cache}"
export XDG_CACHE_HOME="$WS/.cache"
export WANDB_PROJECT=edit-diffusion
export WANDB_MODE=online
export WANDB_DIR="$ROOT/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
cd "$ROOT"
echo "RUN_MARKER host=$(hostname) iso=$(date --iso-8601=seconds) cuda=$($PY -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "RUN_MARKER train_start iso=$(date --iso-8601=seconds)"
"$PY" train.py \
  --dataset gsm8k \
  --steps 20000 \
  --batch-size 64 \
  --grad-accum 8 \
  --lr 3e-4 \
  --weight-decay 0.03 \
  --warmup-steps 500 \
  --hidden-dim 768 \
  --num-layers 12 \
  --num-heads 12 \
  --dropout 0.1 \
  --max-prompt-len 640 \
  --max-target-len 768 \
  --coupling empty \
  --val-every 2000 \
  --patience 5 \
  --grad-clip 1.0 \
  --t-eps 1e-2 \
  --precision bf16 \
  --task-eval-samples 16 \
  --task-eval-steps 128 \
  --val-max-examples 256 \
  --wandb-project edit-diffusion \
  --wandb-name editflow-gsm8k-v4 \
  --save-dir "$WS/scratch/editflow-gsm8k-v4" \
  --save-every 2000 \
  --log-every 20 \
  --seed 42
echo "RUN_MARKER train_end iso=$(date --iso-8601=seconds)"
BEST="$WS/scratch/editflow-gsm8k-v4/model_best.pt"
if [ -f "$BEST" ]; then
  echo "RUN_MARKER eval_start iso=$(date --iso-8601=seconds)"
  "$PY" sample.py --checkpoint "$BEST" --num-prompts 64 --num-steps 256 --batch-size 8 \
    --out "$WS/scratch/editflow-gsm8k-v4/samples.json"
  echo "RUN_MARKER eval_end iso=$(date --iso-8601=seconds)"
fi
echo "RUN_MARKER end iso=$(date --iso-8601=seconds) status=ok"
