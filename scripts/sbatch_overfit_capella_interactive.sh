#!/usr/bin/env bash
# 1-GPU TinyGSM 5-example overfit on Capella debug MIG. Sequential MDLM→FlexMDM→PUMA→EditFlow.
#SBATCH --job-name=overfit5-tinygsm
#SBATCH --account=p_scads_ddlm
#SBATCH --partition=capella-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/data/horse/ws/alpo658h-insert-diff/projects/bayesian-group-research/outputs/overfit/slurm-%j.out

set -euo pipefail
REPO=/data/horse/ws/alpo658h-insert-diff/projects/bayesian-group-research
mkdir -p "$REPO/outputs/overfit"
cd "$REPO"
bash scripts/run_overfit.sh
