#!/usr/bin/env bash
# 1-node x 4-GPU FlexMDM on an existing Capella holder (srun --overlap).
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
if [ -n "${VIRTUAL_ENV:-}" ]; then
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  hash -r
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  true
elif [ -f "$REPO_ROOT/scripts/env.sh" ] && [ -n "${CONDA_PREFIX_OVERRIDE:-}${CONDA_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/env.sh"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-edit-diffusion}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HYDRA_FULL_ERROR=1
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT WANDB_ENTITY || true

NPROC="${NPROC_PER_NODE:-4}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/wandb}"
mkdir -p "$SAVE_ROOT" "$WANDB_DIR" "${PRETOKENIZED_ROOT:?set PRETOKENIZED_ROOT}"

if [ ! -s "$PRETOKENIZED_ROOT/manifest.json" ]; then
  echo "RUN_MARKER tokenize_start iso=$(date --iso-8601=seconds)"
  python -m flexmdm.data tokenize \
    --config flexmdm/config/fsdp_train.yaml \
    --output-root "$PRETOKENIZED_ROOT"
  python scripts/drop_truncated_rows.py --root "$PRETOKENIZED_ROOT"
  python scripts/merge_manifests.py --root "$PRETOKENIZED_ROOT"
  echo "RUN_MARKER tokenize_end iso=$(date --iso-8601=seconds)"
else
  echo "RUN_MARKER tokenize_skip reason=manifest_exists"
fi

echo "RUN_MARKER train_start iso=$(date --iso-8601=seconds) nproc=$NPROC"
python -m torch.distributed.run \
  --standalone --nnodes 1 --nproc_per_node "$NPROC" \
  -m flexmdm.train_fsdp \
  --config-name fsdp_train \
  "trainer.default_local_dir=$SAVE_ROOT" \
  "trainer.wandb.dir=$WANDB_DIR" \
  "trainer.project_name=$WANDB_PROJECT" \
  "trainer.experiment_name=${WANDB_NAME:-genuine-flexmdm-capella}" \
  "trainer.wandb.mode=online" \
  "trainer.early_stop_patience=5" \
  "fsdp.sharding_strategy=full_shard" \
  "model.attn_implementation=flash_attention_2" \
  "model.enable_gradient_checkpointing=true"
echo "RUN_MARKER train_end iso=$(date --iso-8601=seconds) status=$?"
