#!/bin/bash
# Vanilla MDM at the puma-tinygsm-256 compute/data/eval recipe (not PUMA progressive).
# 4 GPUs, global batch 128, full TinyGSM L=256, 260k steps, GSM8K Python-exec eval.
set -euo pipefail

PUMA_DIR="${PUMA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
NPROC="${NPROC:-4}"
CFG="${CFG:-yaml_files/tinygsm_mdm_256.yaml}"
DATA_DIR="data/tiny_gsm_256"

cd "$PUMA_DIR"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  hash -r
fi

echo "RUN_MARKER launch_start iso=$(date --iso-8601=seconds) host=$(hostname) python=$(command -v python)"

if [ -s "$DATA_DIR/meta.json" ]; then
  echo "RUN_MARKER preprocess_skip reason=tinygsm_cache_exists dir=$DATA_DIR"
else
  echo "RUN_MARKER preprocess_start iso=$(date --iso-8601=seconds)"
  python data/tiny_gsm.py --out_dir "$DATA_DIR" --max_len 256 --num_proc "${SLURM_CPUS_PER_TASK:-16}"
  echo "RUN_MARKER preprocess_end iso=$(date --iso-8601=seconds)"
fi
if [ -s "data/gsm8k_test/test_mdm_256.json" ]; then
  echo "RUN_MARKER gsm8k_cache_skip reason=exists"
else
  echo "RUN_MARKER gsm8k_cache_start iso=$(date --iso-8601=seconds)"
  python -c "from eval.gsm8k_eval import test_gsm8k_tokenization, MASK_ID; test_gsm8k_tokenization(MASK_ID, max_len=256)"
  echo "RUN_MARKER gsm8k_cache_end iso=$(date --iso-8601=seconds)"
fi

echo "RUN_MARKER train_start iso=$(date --iso-8601=seconds) nproc=$NPROC"
if [ "$NPROC" -eq 1 ]; then
  python train.py --cfg "$CFG"
else
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    train.py --cfg "$CFG"
fi
status=$?
echo "RUN_MARKER train_end iso=$(date --iso-8601=seconds) status=$status"
exit "$status"
