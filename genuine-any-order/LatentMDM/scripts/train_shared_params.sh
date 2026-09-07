#!/usr/bin/env bash
# TinyGSM 100k, planner L=256, global batch 64, 20k steps (srun --overlap).
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
if [ -n "${VIRTUAL_ENV:-}" ]; then
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  hash -r
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-edit-diffusion}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT WANDB_ENTITY || true

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/tiny_gsm_shared_params}"
NPROC="${NPROC_PER_NODE:-4}"
GLOBAL_BS="${GLOBAL_BS:-64}"
PER_GPU_BS="${PER_GPU_BS:-$((GLOBAL_BS / NPROC))}"
if [ "$((PER_GPU_BS * NPROC))" -ne "$GLOBAL_BS" ]; then
  echo "GLOBAL_BS=$GLOBAL_BS is not divisible by NPROC=$NPROC" >&2
  exit 1
fi

if [ ! -s "$DATA_DIR/meta.json" ]; then
  echo "RUN_MARKER preprocess_start iso=$(date --iso-8601=seconds)"
  mkdir -p "$DATA_DIR"
  TINYGSM_OUT_DIR="$DATA_DIR" \
    TINYGSM_LIMIT=100000 \
    TINYGSM_MAX_LEN=256 \
    TINYGSM_MAX_PROMPT_LEN=128 \
    python data/tiny_gsm.py
  echo "RUN_MARKER preprocess_end iso=$(date --iso-8601=seconds)"
else
  echo "RUN_MARKER preprocess_skip reason=dataset_exists"
fi

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  RESUME_ARGS+=(--resume "$RESUME")
fi
echo "RUN_MARKER train_start iso=$(date --iso-8601=seconds) nproc=$NPROC per_gpu_bs=$PER_GPU_BS python=$(command -v python) resume=${RESUME:-} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
if [ "$NPROC" -eq 1 ]; then
  python train.py --cfg yaml_files/tinygsm_shared_params.yaml \
    data.data_dir="$DATA_DIR" \
    data.training.per_gpu_batch_size="$PER_GPU_BS" \
    training.batch_size="$PER_GPU_BS" \
    wandb.project="$WANDB_PROJECT" \
    wandb.name="${WANDB_NAME:-latentmdm-tinygsm-L256-shared-params}" \
    wandb.entity=null \
    wandb.wandb=true \
    "${RESUME_ARGS[@]}"
else
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    train.py --cfg yaml_files/tinygsm_shared_params.yaml \
    data.data_dir="$DATA_DIR" \
    data.training.per_gpu_batch_size="$PER_GPU_BS" \
    training.batch_size="$PER_GPU_BS" \
    wandb.project="$WANDB_PROJECT" \
    wandb.name="${WANDB_NAME:-latentmdm-tinygsm-L256-shared-params}" \
    wandb.entity=null \
    wandb.wandb=true \
    "${RESUME_ARGS[@]}"
fi
status=$?
echo "RUN_MARKER train_end iso=$(date --iso-8601=seconds) status=$status"
if [ "$status" -eq 0 ]; then
  CKPT=$(ls -t ckpts/date=*/step=*.pt 2>/dev/null | head -1 || true)
  if [ -n "$CKPT" ]; then
    echo "RUN_MARKER eval_start ckpt=$CKPT iso=$(date --iso-8601=seconds)"
    python eval.py --cfg yaml_files/tinygsm_shared_params.yaml --resume "$CKPT" \
      validation.limit=none \
      wandb.wandb=true \
      wandb.project="$WANDB_PROJECT" \
      wandb.name="${WANDB_NAME:-latentmdm-tinygsm-L256-shared-params}-eval" \
      wandb.entity=null \
      || echo "RUN_MARKER eval_failed"
    echo "RUN_MARKER eval_end iso=$(date --iso-8601=seconds)"
  else
    echo "RUN_MARKER eval_skip reason=no_checkpoint"
  fi
fi
exit "$status"
