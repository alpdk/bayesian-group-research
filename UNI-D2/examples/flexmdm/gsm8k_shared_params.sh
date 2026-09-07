#!/usr/bin/env bash
# GSM8K FlexMDM shared-params. Knobs: configs/flexmdm/gsm8k.yaml
set -euo pipefail

UNI_D2_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMPARISON_ROOT="$(cd "${UNI_D2_ROOT}/.." && pwd)"
CFG="${CFG:-${COMPARISON_ROOT}/configs/flexmdm/gsm8k.yaml}"
cd "${UNI_D2_ROOT}"

export PYTHONPATH="${UNI_D2_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mapfile -t OVERRIDES < <(python "${COMPARISON_ROOT}/scripts/hydra_from_yaml.py" "${CFG}")
python -u -m discrete_diffusion "${OVERRIDES[@]}" "$@"
