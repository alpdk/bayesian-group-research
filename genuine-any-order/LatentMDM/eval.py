"""Evaluation entry point for ARM, MDM, and LatentMDM."""

import argparse

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from eval.runner import evaluate_ddp_dict
from model.ema import ExponentialMovingAverage
from model.factory import COMBINED_STRATEGIES, build_model
from utils import seed_everything, setup_ddp as _setup_ddp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to evaluate",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser.parse_args()


def setup_ddp():
    """Initialize evaluation DDP with PyTorch's default timeout."""
    return _setup_ddp()


def main(cfg: DictConfig, resume_path: str = None):
    if resume_path is None:
        raise ValueError("eval.py requires --resume to point to a checkpoint.")

    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start evaluation!")
        print(f"Evaluating with {world_size} GPUs")
    
    seed_everything(rank)

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    train_cfg = cfg.training
    val_cfg = cfg.validation
    strategy = train_cfg.strategy
    wandb_initialized = False

    try:
        # Initialize the model
        model, model_config = build_model(cfg, device, is_main=is_main)

        if is_main:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model is ready, parameters: {num_params/1e6:.2f}M")

        # model wrapping
        if world_size > 1 and torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            if is_main:
                print(f"Model wrapping is done!")

        eos_id = getattr(val_cfg.sampling, "eos_id", None)
        if eos_id is None:
            eos_id = getattr(train_cfg, "eos_id", None)
        if strategy in COMBINED_STRATEGIES and eos_id is None:
            raise ValueError("LatentMDM evaluation requires validation.sampling.eos_id or training.eos_id.")

        if is_main:
            print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint is missing required key 'model_state_dict': {resume_path}")

        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"], strict=True)
        global_step = ckpt.get("global_step", 0)
        epoch = ckpt.get("epoch", 0)

        if train_cfg.ema is not None:
            if "ema_state_dict" not in ckpt:
                raise KeyError(
                    "training.ema is enabled, but checkpoint is missing "
                    f"required key 'ema_state_dict': {resume_path}"
                )
            assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
            ema_params = [p for p in model_to_load.parameters() if p.requires_grad]
            ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.move_shadow_params_to_device(device)
            ema.copy_to(model_to_load.parameters())
            if is_main:
                print("Loaded EMA weights from checkpoint.")
        elif is_main:
            print("Loaded model weights from checkpoint.")

        if is_main:
            print(f"Checkpoint metadata: epoch {epoch}, global_step {global_step}")
        if cfg.validation.get("eval_itr", None) is None:
            cfg.validation.eval_itr = int(global_step)

        if cfg.wandb.wandb and is_main:
            wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity)
            wandb_initialized = True

        model.eval()
        with torch.inference_mode():
            val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)

        if is_main:
            for key, value in val_acc_dict.items():
                print(f"Validation Accuracy {key}: {value}")
            from eval_protocol import extract_task_exact_match
            task_metrics = extract_task_exact_match(val_acc_dict)
            if cfg.wandb.wandb and task_metrics:
                wandb.log(task_metrics, step=global_step)
                wandb.summary.update(task_metrics)

        return val_acc_dict
    finally:
        if wandb_initialized:
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
