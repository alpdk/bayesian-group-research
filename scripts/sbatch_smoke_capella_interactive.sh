#!/usr/bin/env bash
# 1-GPU TinyGSM smoke on Capella debug MIG. Do not use a 7-day holder for this.
#SBATCH --job-name=smoke-tinygsm
#SBATCH --account=p_scads_ddlm
#SBATCH --partition=capella-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/data/horse/ws/alpo658h-insert-diff/projects/bayesian-group-research/outputs/smoke/slurm-%j.out

set -euo pipefail
REPO=/data/horse/ws/alpo658h-insert-diff/projects/bayesian-group-research
mkdir -p "$REPO/outputs/smoke"
cd "$REPO"
bash scripts/run_smoke.sh
