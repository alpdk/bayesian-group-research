#!/bin/bash
# Launch PUMA TACO training at sequence length 512.
set -euo pipefail
PUMA_DIR="${PUMA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
UNI_D2="$(cd "$PUMA_DIR/.." && pwd)"
VENV="${VENV:-$UNI_D2/.venv}"
NPROC="${NPROC:-1}"
CFG="yaml_files/taco_puma_512.yaml"
DATA_DIR="data/taco_512"
SCRATCH="${DISCRETE_DIFFUSION_SCRATCH_DIR:-$HOME/.cache/discrete_diffusion}"
SLIM="$SCRATCH/taco/taco_raw.dat"

cd "$PUMA_DIR"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
export PYTHONPATH="${UNI_D2}/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -s "$SLIM/dataset_dict.json" ] && [ ! -d "$SLIM" ]; then
  python -c "from discrete_diffusion.data.loaders import load_code_qa_dataset; load_code_qa_dataset('taco', '$SCRATCH/taco')"
fi
if [ ! -s "$DATA_DIR/meta.json" ]; then
  python data/code_qa.py --out_dir "$DATA_DIR" --source_dir "$SLIM" --max_len 512
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py --cfg "$CFG" ${RESUME:+--resume "$RESUME"}
