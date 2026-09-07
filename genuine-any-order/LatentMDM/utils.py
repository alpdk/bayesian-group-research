"""Runtime, distributed, and validation utilities."""

import datetime
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.factory import COMBINED_STRATEGIES
from training.losses import arm_loss, get_prompt_len, latmdm_loss, mdm_loss


def setup_ddp(
    timeout: Optional[datetime.timedelta] = None,
) -> tuple[int, int, int]:
    """Initialize single- or multi-process execution and return rank metadata."""
    if (
        torch.cuda.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    ):
        kwargs = {"backend": "nccl"}
        if timeout is not None:
            kwargs["timeout"] = timeout
        dist.init_process_group(**kwargs)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def seed_everything(rank: int, base_seed: int = 2026) -> int:
    """Seed Python, NumPy, and PyTorch using the process rank."""
    seed = int(base_seed) + int(rank)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def val_loss_ddp(
    model,
    val_loader,
    mask_id: int,
    device,
    rank: int,
    world_size: int,
    strategy: str,
    eos_id: int,
    arm_init: bool = False,
    decoder_chunk_size: int = 1024,
    slot_reweight: bool = False,
) -> float:
    """Compute example-weighted validation loss across distributed ranks."""
    model.eval()
    if (
        world_size > 1
        and dist.is_initialized()
        and not isinstance(val_loader.sampler, DistributedSampler)
    ):
        sampler = DistributedSampler(
            val_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size or 16,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    local_sum = 0.0
    local_count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", disable=(rank != 0)):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                if strategy in COMBINED_STRATEGIES:
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
                    batch_size = prompt.shape[0]
                elif strategy == "arm":
                    labels = batch["labels"].to(device)
                    prompt_mask = (
                        batch["prompt_mask"].to(device)
                        if "prompt_mask" in batch
                        else None
                    )
                    batch_size = labels.shape[0]
                    loss = arm_loss(
                        model,
                        labels,
                        eos_id=eos_id,
                        prompt_mask=prompt_mask,
                    )
                elif strategy == "standard":
                    labels = batch["labels"].to(device)
                    prompt_mask = (
                        batch["prompt_mask"].to(device)
                        if "prompt_mask" in batch
                        else None
                    )
                    batch_size = labels.shape[0]
                    loss = mdm_loss(
                        model,
                        labels,
                        mask_id,
                        prompt_mask=prompt_mask,
                        arm_init=arm_init,
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
            local_sum += float(loss.item() * batch_size)
            local_count += batch_size

    tensor = torch.tensor(
        [local_sum, local_count],
        dtype=torch.float,
        device=device,
    )
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_sum, global_count = tensor.tolist()
    return global_sum / max(int(global_count), 1)


__all__ = ["seed_everything", "setup_ddp", "val_loss_ddp"]
