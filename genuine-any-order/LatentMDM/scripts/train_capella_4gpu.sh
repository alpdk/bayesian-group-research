#!/usr/bin/env bash
# 1-node x 4-GPU LatentMDM on an existing Capella holder (srun --overlap).
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

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/tiny_gsm}"
NPROC="${NPROC_PER_NODE:-4}"
PER_GPU_BS="${PER_GPU_BS:-64}"

if [ ! -s "$DATA_DIR/meta.json" ]; then
  echo "RUN_MARKER preprocess_start iso=$(date --iso-8601=seconds)"
  mkdir -p "$DATA_DIR"
  python data/tiny_gsm.py
  echo "RUN_MARKER preprocess_end iso=$(date --iso-8601=seconds)"
else
  echo "RUN_MARKER preprocess_skip reason=dataset_exists"
fi

echo "RUN_MARKER train_start iso=$(date --iso-8601=seconds) nproc=$NPROC python=$(command -v python)"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  train.py --cfg yaml_files/tinygsm_latentmdm.yaml \
  data.data_dir="$DATA_DIR" \
  data.training.per_gpu_batch_size="$PER_GPU_BS" \
  wandb.project="$WANDB_PROJECT" \
  wandb.name="${WANDB_NAME:-latentmdm-tinygsm-capella}" \
  wandb.entity=null \
  wandb.wandb=true \
  training.early_stop_patience=5
echo "RUN_MARKER train_end iso=$(date --iso-8601=seconds) status=$?"
