#!/usr/bin/env bash
# TACO, planner L=512, global batch 64.
set -euo pipefail
export PYTHONUNBUFFERED=1
if [ -n "${VIRTUAL_ENV:-}" ]; then
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  hash -r
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNI_D2="$(cd "$REPO_ROOT/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${UNI_D2}/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-edit-diffusion}"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/taco_shared_params}"
SCRATCH="${DISCRETE_DIFFUSION_SCRATCH_DIR:-$HOME/.cache/discrete_diffusion}"
SLIM="$SCRATCH/taco/taco_raw.dat"
NPROC="${NPROC_PER_NODE:-1}"
GLOBAL_BS="${GLOBAL_BS:-64}"
PER_GPU_BS="${PER_GPU_BS:-$((GLOBAL_BS / NPROC))}"

if [ ! -d "$SLIM" ] && [ ! -f "$SLIM/dataset_dict.json" ]; then
  python -c "from discrete_diffusion.data.loaders import load_code_qa_dataset; load_code_qa_dataset('taco', '$SCRATCH/taco')"
fi
if [ ! -s "$DATA_DIR/meta.json" ]; then
  mkdir -p "$DATA_DIR"
  TINYGSM_OUT_DIR="$DATA_DIR" \
    TINYGSM_SOURCE=taco \
    TINYGSM_HF_DATASET="$SLIM" \
    TINYGSM_MAX_LEN=512 \
    TINYGSM_MAX_PROMPT_LEN=256 \
    TINYGSM_MAX_SEG_LEN=32 \
    TINYGSM_MAX_SEG_NUM=16 \
    TINYGSM_DROP_OVERFLOW=0 \
    python data/tiny_gsm.py
fi

if [ "$NPROC" -eq 1 ]; then
  python train.py --cfg yaml_files/taco_shared_params.yaml \
    data.data_dir="$DATA_DIR" \
    data.training.per_gpu_batch_size="$PER_GPU_BS" \
    training.batch_size="$PER_GPU_BS" \
    wandb.project="$WANDB_PROJECT" \
    wandb.name="${WANDB_NAME:-latentmdm-taco-L512-shared-params}" \
    wandb.entity=null \
    wandb.wandb=true
else
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    train.py --cfg yaml_files/taco_shared_params.yaml \
    data.data_dir="$DATA_DIR" \
    data.training.per_gpu_batch_size="$PER_GPU_BS" \
    training.batch_size="$PER_GPU_BS" \
    wandb.project="$WANDB_PROJECT" \
    wandb.name="${WANDB_NAME:-latentmdm-taco-L512-shared-params}" \
    wandb.entity=null \
    wandb.wandb=true
fi
