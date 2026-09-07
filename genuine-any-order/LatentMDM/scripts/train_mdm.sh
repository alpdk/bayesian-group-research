#!/usr/bin/env bash
#SBATCH --job-name tinygsm_rep

### Logging
#SBATCH --output job_logs/train_%j.out
#SBATCH --error  job_logs/train_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=sgkim@utexas.edu

### Node info
#SBATCH --account ASC25024
#SBATCH --nodes 1
#SBATCH --partition gh
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00

source ~/.bashrc
micromamba activate LatentMDM

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun -N1 -n1 -w "$MASTER_HOST" hostname -I | awk '{print $1}')
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export HF_HOME="/scratch/10816/sk58348/vista/hf_cache"
export XDG_CACHE_HOME="/scratch/10816/sk58348/vista/xdg_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hardcoded OmegaConf dotlist overrides — uncomment / edit as needed.
# CLI args passed after CFG_PATH override these (later entries win).
OVERRIDES=(
  wandb.name="train_mdm_125M_ema"
)

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train.py --cfg yaml_files/tinygsm_mdm.yaml "${OVERRIDES[@]}" "$@" # --resume "$CHECKPOINT_PATH"
