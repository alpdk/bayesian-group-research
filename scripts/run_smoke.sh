#!/usr/bin/env bash
# TinyGSM smoke: 80 packed examples, 20 optimizer steps, 1 GPU.
# Checks that each trainer takes a finite train/loss (no generate+score).
#
#   METHODS="mdlm flexmdm puma editflow" bash scripts/run_smoke.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
WS="${WORKSPACE:-/data/horse/ws/alpo658h-insert-diff}"
PY="${PYTHON:-$WS/conda_envs/uni-d2/bin/python}"
METHODS="${METHODS:-mdlm flexmdm puma editflow}"
LOG_DIR="${LOG_DIR:-$REPO/outputs/smoke}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$LOG_DIR}"
export HF_HOME="${HF_HOME:-$WS/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WS/.cache}"
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DISCRETE_DIFFUSION_SCRATCH_DIR:-$WS/scratch/discrete_diffusion}"
export PATH="$(dirname "$PY"):${PATH}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

mkdir -p "$LOG_DIR"
echo "RUN_MARKER smoke_start host=$(hostname) iso=$(date --iso-8601=seconds) py=$PY cuda=$($PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)')"

run_mdlm() {
  echo "RUN_MARKER method=mdlm start iso=$(date --iso-8601=seconds)"
  cd "$REPO/UNI-D2"
  export PYTHONPATH="$REPO/UNI-D2/src"
  mapfile -t OVERRIDES < <("$PY" "$REPO/scripts/hydra_from_yaml.py" "$REPO/configs/smoke/mdlm_tinygsm.yaml")
  "$PY" -u -m discrete_diffusion "${OVERRIDES[@]}" || return 1
  echo "RUN_MARKER method=mdlm end iso=$(date --iso-8601=seconds)"
}

run_flexmdm() {
  echo "RUN_MARKER method=flexmdm start iso=$(date --iso-8601=seconds)"
  cd "$REPO/UNI-D2"
  export PYTHONPATH="$REPO/UNI-D2/src"
  mapfile -t OVERRIDES < <("$PY" "$REPO/scripts/hydra_from_yaml.py" "$REPO/configs/smoke/flexmdm_tinygsm.yaml")
  "$PY" -u -m discrete_diffusion "${OVERRIDES[@]}" || return 1
  echo "RUN_MARKER method=flexmdm end iso=$(date --iso-8601=seconds)"
}

run_puma() {
  echo "RUN_MARKER method=puma start iso=$(date --iso-8601=seconds)"
  cd "$REPO/PUMA"
  unset PYTHONPATH
  DATA_DIR="data/tiny_gsm_256_smoke"
  if [ ! -s "$DATA_DIR/meta.json" ]; then
    echo "RUN_MARKER puma pretok start"
    "$PY" data/tiny_gsm.py --out_dir "$DATA_DIR" --max_len 256 --limit 80 --num_proc 1 || return 1
  fi
  "$PY" train.py --cfg "$REPO/configs/smoke/puma_tinygsm.yaml" || return 1
  echo "RUN_MARKER method=puma end iso=$(date --iso-8601=seconds)"
}

run_editflow() {
  echo "RUN_MARKER method=editflow start iso=$(date --iso-8601=seconds)"
  cd "$REPO/edit_flows_code"
  unset PYTHONPATH
  SAVE_DIR="${SAVE_DIR_EDITFLOW:-$REPO/outputs/smoke/editflow-tinygsm}"
  mkdir -p "$SAVE_DIR"
  "$PY" train.py --cfg "$REPO/configs/smoke/editflow_tinygsm.yaml" --save-dir "$SAVE_DIR" || return 1
  echo "RUN_MARKER method=editflow end iso=$(date --iso-8601=seconds)"
}

failed=0
for method in $METHODS; do
  case "$method" in
    mdlm) run_mdlm || { echo "RUN_MARKER method=mdlm FAILED"; failed=1; } ;;
    flexmdm) run_flexmdm || { echo "RUN_MARKER method=flexmdm FAILED"; failed=1; } ;;
    puma) run_puma || { echo "RUN_MARKER method=puma FAILED"; failed=1; } ;;
    editflow) run_editflow || { echo "RUN_MARKER method=editflow FAILED"; failed=1; } ;;
    *) echo "unknown method: $method" >&2; exit 2 ;;
  esac
done

echo "RUN_MARKER smoke_end iso=$(date --iso-8601=seconds) failed=$failed"
exit "$failed"
