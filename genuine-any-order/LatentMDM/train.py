"""Training entry point for ARM, MDM, and LatentMDM."""

import argparse
import datetime
import math
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data import setup_data_bundle
from eval.runner import evaluate_ddp, evaluate_ddp_dict
from eval_protocol import (
    extract_task_exact_match,
    nll_family,
    update_best_summary,
    update_early_stop,
)
from model.ema import ExponentialMovingAverage, save_model_snapshot
from model.factory import COMBINED_STRATEGIES, build_model
from training.losses import arm_loss, get_prompt_len, grad_norm, latmdm_loss, mdm_loss
from utils import seed_everything, setup_ddp as _setup_ddp, val_loss_ddp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args, extras = parser.parse_known_args()
    args.overrides = extras
    return args


def setup_ddp():
    """Initialize training DDP with the historical one-hour timeout."""
    return _setup_ddp(timeout=datetime.timedelta(hours=1))


def main(cfg: DictConfig, resume_path: str = None):
    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start training!")
        print(f"Training with {world_size} GPUs")
    
    seed_everything(rank, base_seed=int(cfg.data.get("seed", 42)))

    # ckpt dir
    ckpt_dir = f"ckpts/date={datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}-{random.SystemRandom().randint(0, 99999):05d}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    data_cfg = cfg.data
    train_cfg = cfg.training
    val_cfg = cfg.validation
    strategy = train_cfg.strategy

    # Initialize the model
    model, model_config = build_model(cfg, device, is_main=is_main)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model is ready, parameters: {num_params/1e6:.2f}M")

    # model wrapping
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main:
            print(f"Model wrapping is done!")

    # data
    assert train_cfg.save_steps % train_cfg.eval_steps == 0, "save_steps must be divisible by eval_steps"
    data_bundle = setup_data_bundle(data_cfg)
    train_loader, val_loader = data_bundle.train_loader, data_bundle.val_loader    
    mask_id = data_cfg.mask_id
    eos_id = getattr(val_cfg.sampling, "eos_id", None)
    if eos_id is None:
        eos_id = getattr(train_cfg, "eos_id", None)
    if strategy in COMBINED_STRATEGIES and eos_id is None:
        raise ValueError("LP-MDM training requires validation.sampling.eos_id or training.eos_id.")
    decoder_chunk_size = int(getattr(train_cfg, "decoder_chunk_size", 1024))
    slot_reweight = bool(getattr(train_cfg, "slot_reweight", False))
    gradient_accumulation_steps = int(getattr(train_cfg, "gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1.")
    if is_main and gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation: {gradient_accumulation_steps} microbatches per optimizer step")

    # training hyperparemeters
    # attach DDP sampler
    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_cfg.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        train_sampler = None

    # optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    num_training_steps = train_cfg.num_epochs * num_update_steps_per_epoch
    max_steps_cfg = int(getattr(train_cfg, "max_steps", 0) or 0)
    if max_steps_cfg > 0:
        num_training_steps = max(num_training_steps, max_steps_cfg)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps)
    if train_cfg.ema is not None:
        assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
        model_to_ema = model.module if isinstance(model, DDP) else model
        ema_params = [p for p in model_to_ema.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
        if is_main:
            print("EMA is enabled with decay:", train_cfg.ema)

    # resume from checkpoint
    start_epoch = 0
    start_global_step = 0
    if resume_path is not None:
        if is_main:
            print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if train_cfg.ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.move_shadow_params_to_device(device)
        start_epoch = ckpt.get("epoch", 0)
        start_global_step = ckpt.get("global_step", 0)
        scheduler.last_epoch = start_global_step
        for param_group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            param_group["lr"] = lr
        if is_main:
            print(f"Resumed at epoch {start_epoch}, global_step {start_global_step}")

    # training loop
    global_step = start_global_step
    last_grad_norm = None
    optimizer.zero_grad(set_to_none=True)
    best_val_nll = None
    stale_val_checks = 0
    early_stop_patience = int(getattr(train_cfg, "early_stop_patience", 5) or 0)
    early_stop_min_delta = float(getattr(train_cfg, "early_stop_min_delta", 0.0) or 0.0)
    stop_training = False

        
    # wandb initialize
    if cfg.wandb.wandb and is_main:
        os.environ["WANDB_PROJECT"] = "edit-diffusion"
        os.environ["WANDB_MODE"] = "online"
        wandb_kwargs = dict(
            project="edit-diffusion",
            name=cfg.wandb.name,
            entity=os.environ.get("WANDB_ENTITY") or None,
            tags=["latentmdm", "genuine-any-order", "tinygsm", "gsm8k", "shared-params", "L256"],
            mode="online",
        )
        run_id = os.environ.get("WANDB_RUN_ID")
        if run_id:
            wandb_kwargs["id"] = run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)

    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        iterable = train_loader

        if is_main:
            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}")
        else:
            pbar = iterable

        # compute how many steps to skip in this epoch when resuming
        steps_in_epoch = len(train_loader)
        skip_steps = (
            max(0, start_global_step * gradient_accumulation_steps - epoch * steps_in_epoch)
            if epoch == start_epoch
            else 0
        )
        accum_count = 0
        accum_window_size = gradient_accumulation_steps

        for step_in_epoch, itr in enumerate(pbar):
            if step_in_epoch < skip_steps:
                continue
            if accum_count == 0:
                remaining_micro_steps = steps_in_epoch - step_in_epoch
                accum_window_size = min(gradient_accumulation_steps, remaining_micro_steps)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                step_t0 = time.perf_counter()
            is_update_step = (accum_count + 1) == accum_window_size

            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not is_update_step
                else nullcontext()
            )
            with sync_context:
                # to enable flashattention, we do the autocast
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                    if strategy == "standard":
                        batch = itr
                        input_ids = batch["labels"].to(device)
                        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                        loss = mdm_loss(model, input_ids, mask_id, prompt_mask = prompt_mask, arm_init=model_config.predict_next_token)
                    elif strategy == "arm":
                        batch = itr
                        input_ids = batch["labels"].to(device)
                        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                        loss = arm_loss(model, input_ids, eos_id=eos_id, prompt_mask=prompt_mask)
                    elif strategy in COMBINED_STRATEGIES:
                        batch = itr
                        prompt = batch["prompt"].to(device)
                        split_labels = batch["split_labels"].to(device)
                        prompt_len = get_prompt_len(batch, device)
                        loss = latmdm_loss(
                            model,
                            prompt,
                            split_labels,
                            prompt_len,
                            eos_id=eos_id,
                            decoder_chunk_size=decoder_chunk_size,
                            slot_reweight=slot_reweight,
                        )
                    else:
                        raise ValueError(f"Invalid training strategy: {strategy}")

                (loss / accum_window_size).backward()
            accum_count += 1

            if is_update_step:
                if train_cfg.max_grad_norm > 0:
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_cfg.max_grad_norm,
                    ).item()
                else:
                    last_grad_norm = grad_norm(model.parameters())
                optimizer.step()
                if train_cfg.ema is not None:
                    ema.update(ema_params)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0
                max_steps = int(getattr(train_cfg, "max_steps", 0) or 0)
                if max_steps > 0 and global_step >= max_steps:
                    stop_training = True

            if is_main:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

                if is_update_step and global_step % train_cfg.logging_steps == 0:
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss {loss.item()}")
                    if cfg.wandb.wandb:
                        step_s = time.perf_counter() - step_t0
                        payload = {
                            "train/loss": float(loss.item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "perf/train_step_s": step_s,
                        }
                        if last_grad_norm is not None:
                            payload["healthy/grad_norm"] = float(last_grad_norm)
                        n_tok = 0
                        if isinstance(itr, dict):
                            for key in ("labels", "prompt", "split_labels"):
                                value = itr.get(key)
                                if value is not None and hasattr(value, "numel"):
                                    n_tok += int(value.numel())
                        if step_s > 0 and n_tok > 0:
                            payload["perf/train_tokens_per_s"] = n_tok * max(world_size, 1) / step_s
                        wandb.log(payload, step=global_step)

            if is_update_step and global_step > 0 and global_step % train_cfg.eval_steps == 0:
                model.eval()
                test_s = 0.0
                val_acc_dict = None
                if train_cfg.ema is None:
                    t_test = time.perf_counter()
                    try:
                        val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                    except Exception as exc:
                        if is_main:
                            print(f"Task eval failed at step {global_step}: {exc}", flush=True)
                        val_acc_dict = None
                    test_s = time.perf_counter() - t_test

                t_val = time.perf_counter()
                val_loss = val_loss_ddp(
                    model,
                    val_loader,
                    mask_id,
                    device,
                    rank,
                    world_size,
                    strategy,
                    eos_id,
                    arm_init=model_config.predict_next_token,
                    decoder_chunk_size=decoder_chunk_size,
                    slot_reweight=slot_reweight,
                )
                val_s = time.perf_counter() - t_val

                if train_cfg.ema is not None:
                    torch.cuda.empty_cache()
                    model_to_ema = model.module if isinstance(model, DDP) else model
                    ema.store(model_to_ema.parameters())
                    ema.copy_to(model_to_ema.parameters())
                    with torch.inference_mode():
                        t_test = time.perf_counter()
                        try:
                            val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                        except Exception as exc:
                            if is_main:
                                print(f"Task eval failed at step {global_step}: {exc}", flush=True)
                            val_acc_dict = None
                        test_s = time.perf_counter() - t_test
                    ema.restore(model_to_ema.parameters())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if is_main:
                    protocol = {**nll_family(float(val_loss)), "perf/val_step_s": val_s}
                    if val_acc_dict is not None:
                        protocol.update(extract_task_exact_match(val_acc_dict))
                        protocol["perf/test_s"] = test_s
                        for key, value in val_acc_dict.items():
                            print(
                                f"Epoch {epoch+1}, Step {global_step}, "
                                f"Validation Accuracy {key}: {value}"
                            )
                    print(f"Epoch {epoch+1}, Step {global_step}, Validation Loss: {val_loss}")
                    if cfg.wandb.wandb:
                        wandb.log(protocol, step=global_step)
                        update_best_summary(wandb.run.summary, protocol)

                    if is_main and global_step % train_cfg.save_steps == 0:
                        # save non-EMA snapshot
                        save_extra = {
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        }
                        if train_cfg.ema is not None:
                            save_extra["ema_state_dict"] = ema.state_dict()
                        if val_acc_dict is not None:
                            save_extra.update(val_acc_dict)
                        saved_path = save_model_snapshot(
                            ckpt_dir, model, cfg, epoch, global_step,
                            val_loss=val_loss,
                            extra=save_extra,
                        )
                        if saved_path is not None:
                            print(f"Model saved to: {saved_path}")

                stop = False
                if early_stop_patience > 0:
                    stop, best_val_nll, stale_val_checks = update_early_stop(
                        float(val_loss),
                        best_val_nll,
                        stale_val_checks,
                        patience=early_stop_patience,
                        min_delta=early_stop_min_delta,
                    )
                if world_size > 1 and dist.is_initialized():
                    flag = torch.tensor(
                        [1 if stop else 0], device=device, dtype=torch.int32
                    )
                    dist.broadcast(flag, src=0)
                    stop = bool(int(flag.item()))
                if stop:
                    if is_main:
                        print(
                            f"Early stopping at step {global_step}: "
                            f"val/nll did not improve for {stale_val_checks} checks "
                            f"(best={best_val_nll}, patience={early_stop_patience})",
                            flush=True,
                        )
                    stop_training = True

                model.train()

            if stop_training:
                break

        if stop_training:
            break

    if cfg.wandb.wandb and is_main:
        wandb.finish()
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg
    cfg = OmegaConf.load(cfg_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg, resume_path=args.resume)
