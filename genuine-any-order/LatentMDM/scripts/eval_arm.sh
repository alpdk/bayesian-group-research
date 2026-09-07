#!/usr/bin/env bash


set -eo pipefail
source ~/.bashrc
set -u
micromamba activate LatentMDM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
export HF_HOME="/scratch/10816/sk58348/vista/hf_cache"
export XDG_CACHE_HOME="/scratch/10816/sk58348/vista/xdg_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
CFG_PATH="${1:-yaml_files/tinygsm_arm.yaml}"
CHECKPOINT_PATH="YOUR_CHECKPOINT_PATH_HERE" # Replace with the actual checkpoint path
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=genuine_any_order_eval \
  eval.py --cfg "$CFG_PATH" --resume "$CHECKPOINT_PATH" "${@:3}"
