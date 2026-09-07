#!/usr/bin/env bash
# TinyGSM 100k, L=256, global batch 64. In-domain TinyGSM exec + GSM8K transfer.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -u -m discrete_diffusion \
    data=tinygsm \
    algo=flexmdm-anyorder \
    model=flexmdm_anyorder \
    sampling=default \
    noise=linear \
    sampling.steps=128 \
    model.length=256 \
    loader.global_batch_size=64 \
    loader.eval_global_batch_size=32 \
    loader.num_workers=4 \
    trainer.max_steps=260000 \
    trainer.val_check_interval=2000 \
    trainer.precision=bf16 \
    trainer.gradient_clip_val=1.0 \
    trainer.log_every_n_steps=100 \
    trainer.num_sanity_val_steps=0 \
    eval.task=tinygsm \
    eval.transfer_task=gsm8k \
    eval.task_num_samples=32 \
    eval.task_checkpoint_monitor='"test/pass@1"' \
    eval.generate_samples=true \
    lr_scheduler=cosine_decay_warmup \
    lr_scheduler.warmup_t=1000 \
    lr_scheduler.warmup_lr_init=1e-6 \
    lr_scheduler.lr_min=0.0 \
    optim.lr=3e-4 \
    optim.weight_decay=0.03 \
    training.ema=0.9999 \
    seed=42 \
    wandb.project=edit-diffusion \
    wandb.entity=alpdk-podkopaev \
    wandb.name=flexmdm-tinygsm-L256-shared-params \
    wandb.group=shared-params \
    wandb.tags=[tinygsm,flexmdm-anyorder,shared-params,L256] \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2000 \
    callbacks.early_stopping.patience=9999 \
    callbacks.sample_saver.enabled=false \
    checkpointing.resume_from_ckpt=true \
    hydra.run.dir=./outputs/tinygsm/flexmdm-shared-params \
    "$@"
