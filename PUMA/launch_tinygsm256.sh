#!/bin/bash
# Launch PUMA TinyGSM training at sequence length 256 with early stopping.
#
# Intended to run on a single GPU node (4 GPUs), either:
#   - inside an existing allocation:
#       srun --jobid=<job-id> --overlap --nodelist=<node> --nodes=1 --ntasks=1 \
#            --gres=gpu:4 --cpus-per-task=32 --mem=256G bash launch_tinygsm256.sh
#   - or from an sbatch script that allocates 1 node with 4 GPUs.
#
# Optional environment overrides:
#   PUMA_DIR   repo root   (default: directory of this script)
#   VENV       python venv (default: $PUMA_DIR/.venv)
#   NPROC      GPUs to use (default: 4)

set -euo pipefail

PUMA_DIR="${PUMA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV="${VENV:-$PUMA_DIR/.venv}"
NPROC="${NPROC:-4}"
CFG="yaml_files/tinygsm_puma_256.yaml"
DATA_DIR="data/tiny_gsm_256"

cd "$PUMA_DIR"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi

echo "RUN_MARKER launch_start_epoch=$(date +%s) iso=$(date --iso-8601=seconds) host=$(hostname)"

NNODES="${SLURM_NNODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"

# 1+2. Data caches: built by node 0 only; other nodes wait for the shared FS cache
if [ "$NODE_RANK" -eq 0 ]; then
  if [ -s "$DATA_DIR/meta.json" ]; then
    echo "RUN_MARKER preprocess_skip_epoch=$(date +%s) reason=tinygsm_cache_exists dir=$DATA_DIR"
  else
    echo "RUN_MARKER preprocess_start_epoch=$(date +%s) iso=$(date --iso-8601=seconds)"
    python data/tiny_gsm.py --out_dir "$DATA_DIR" --max_len 256 --num_proc "${SLURM_CPUS_PER_TASK:-16}"
    echo "RUN_MARKER preprocess_end_epoch=$(date +%s) iso=$(date --iso-8601=seconds)"
  fi
  if [ -s "data/gsm8k_test/test_mdm_256.json" ]; then
    echo "RUN_MARKER gsm8k_cache_skip_epoch=$(date +%s) reason=exists"
  else
    echo "RUN_MARKER gsm8k_cache_start_epoch=$(date +%s)"
    python -c "from eval.gsm8k_eval import test_gsm8k_tokenization, MASK_ID; test_gsm8k_tokenization(MASK_ID, max_len=256)"
    echo "RUN_MARKER gsm8k_cache_end_epoch=$(date +%s)"
  fi
else
  echo "RUN_MARKER node_rank=$NODE_RANK waiting_for_data_cache"
  while [ ! -s "$DATA_DIR/meta.json" ] || [ ! -s "data/gsm8k_test/test_mdm_256.json" ]; do
    sleep 10
  done
fi

# 3. Training
echo "RUN_MARKER train_start_epoch=$(date +%s) iso=$(date --iso-8601=seconds) node_rank=$NODE_RANK nnodes=$NNODES host=$(hostname)"
if [ "$NNODES" -gt 1 ]; then
  NODELIST="${SLURM_STEP_NODELIST:-$SLURM_JOB_NODELIST}"
  MASTER_ADDR=$(scontrol show hostnames "$NODELIST" | head -n 1)
  MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
  echo "RUN_MARKER rdzv master=$MASTER_ADDR:$MASTER_PORT node_rank=$NODE_RANK"
  torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="${SLURM_JOB_ID:-puma256}" train.py --cfg "$CFG" ${RESUME:+--resume "$RESUME"}
else
  torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py --cfg "$CFG" ${RESUME:+--resume "$RESUME"}
fi
status=$?
echo "RUN_MARKER train_end_epoch=$(date +%s) iso=$(date --iso-8601=seconds) status=$status node_rank=$NODE_RANK"
