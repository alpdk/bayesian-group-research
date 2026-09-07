"""APPS / TACO hidden-test eval for PUMA (in-domain pass@1)."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from tqdm import tqdm

from eval.gsm8k_eval import get_sep_ids, get_tokenizer
from sampling import mdm_sampling, mdm_sampling_block, arm_sampling

_UNI_D2_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_UNI_D2_SRC) not in sys.path:
    sys.path.insert(0, str(_UNI_D2_SRC))

from discrete_diffusion.data.loaders import load_code_qa_dataset
from discrete_diffusion.evaluations.code_exec import score_apps_record


def _cache_dir(cfg, name: str) -> str:
    explicit = getattr(cfg.data, "code_qa_cache", None)
    if explicit not in (None, "", "null", "none"):
        return str(explicit)
    scratch = os.environ.get(
        "DISCRETE_DIFFUSION_SCRATCH_DIR",
        str(Path.home() / ".cache" / "discrete_diffusion"),
    )
    return os.path.join(scratch, name)


def test_code_qa_tokenization(cfg, mask_id: int, max_len: int, limit: int = 0):
    name = str(cfg.data.dataset)
    cache_dir = _cache_dir(cfg, name)
    ds = load_code_qa_dataset(name, cache_dir)["test"]
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    tokenizer = get_tokenizer()
    sep_ids = get_sep_ids()
    records = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        q_ids = tokenizer(q, add_special_tokens=False).input_ids
        prompt_ids = q_ids + sep_ids
        if len(prompt_ids) >= max_len:
            prompt_ids = prompt_ids[: max_len - 1]
        ids = prompt_ids + [mask_id] * (max_len - len(prompt_ids))
        records.append({"input_ids": ids, "gold": ex.get("input_output") or ""})
    X = np.array([r["input_ids"] for r in records], dtype=np.int64)
    golds = [r["gold"] for r in records]
    return X, golds


def evaluate_ddp_code_qa(model, cfg, device, rank: int, world_size: int, sampling):
    mask_id = cfg.data.mask_id
    diffusion_layout = getattr(cfg.training, "diffusion_layout", None)
    max_len = int(cfg.model.max_position)
    limit = int(getattr(cfg.validation, "limit", 0) or 0)
    X, golds = test_code_qa_tokenization(cfg, mask_id, max_len, limit=limit)
    n_val = len(X)
    per_rank = math.ceil(n_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, n_val)
    batch_size = 16
    tokenizer = get_tokenizer()
    local_correct, local_total = 0, 0
    with torch.no_grad():
        for s in tqdm(range(start, end, batch_size), desc="Evaluating", disable=(rank != 0)):
            e = min(s + batch_size, end)
            batch_X = torch.from_numpy(X[s:e]).long().to(device)
            if diffusion_layout == "block":
                samples_tensor = mdm_sampling_block(
                    model, batch_X, cfg.training.block_size, mask_id, sampling, device)
            elif cfg.training.strategy == "arm":
                samples_tensor = arm_sampling(model, batch_X, mask_id, sampling, device)
            else:
                samples_tensor = mdm_sampling(
                    model, batch_X, mask_id, sampling, device,
                    arm_init=cfg.model.arm_init != "none")
            samples_tensor = samples_tensor.masked_fill(
                samples_tensor == mask_id, tokenizer.pad_token_id)
            samples = tokenizer.batch_decode(
                samples_tensor.cpu().numpy(), skip_special_tokens=True)
            for sample, gold in zip(samples, golds[s:e]):
                scored = score_apps_record(sample, gold)
                if scored.get("correct"):
                    local_correct += 1
                local_total += 1
    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()
    return global_correct / max(global_total, 1)
