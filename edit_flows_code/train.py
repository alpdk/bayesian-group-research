"""Train a prompt-conditional Edit Flows model for code generation.

Examples (from the edit_flows_code/ directory):
    python train.py --dataset gsm8k --steps 3000
    python train.py --dataset tinygsm --max-examples 100000 --steps 10000
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval_task import (
    aggregate_task_metrics,
    score_gsm8k_python_record,
    score_gsm8k_record,
)
try:
    from eval_task import score_apps_record
except ImportError:  # older eval_task.py without APPS/TACO helpers
    score_apps_record = None

from align import (
    fill_gap_tokens_with_repeats,
    make_ut_mask_from_z,
    opt_align_xs_to_zs,
    rm_gap_tokens,
)
from data import DATASET_CHOICES, CodeGenDataset, load_eval_pairs, load_pairs, pad_stack
from flow import (
    Coupling,
    CubicScheduler,
    EmptyCoupling,
    ExtendedCoupling,
    KappaScheduler,
    UniformCoupling,
    sample_p,
    x2prob,
)
from model import CondEditFlowsTransformer, load_checkpoint, save_checkpoint
from tokenizer import ByteTokenizer


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


def sample_cond_pt(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, kappa: KappaScheduler):
    t = t.reshape(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)


def make_training_batch(
    prompt_list: list[torch.Tensor],
    target_list: list[torch.Tensor],
    coupling: Coupling,
    tok: ByteTokenizer,
):
    """Couples, aligns, pads and BOS-prefixes a batch of variable-length pairs.

    Returns prompt tensor + pad mask, and z_0 / z_1 aligned in Z-space.
    """
    z0s, z1s = [], []
    for x1 in target_list:
        x1b = x1.unsqueeze(0)
        x0b, _ = coupling.sample(x1b)
        # Couplings may emit pad tokens (e.g. UniformCoupling); drop them
        x0b = x0b[:, x0b[0] != tok.pad_token] if x0b.numel() else x0b
        z0b, z1b = opt_align_xs_to_zs(x0b, x1b, gap_token=tok.gap_token)
        z0s.append(z0b[0])
        z1s.append(z1b[0])

    z_0 = pad_stack(z0s, tok.pad_token)
    z_1 = pad_stack(z1s, tok.pad_token)

    # BOS anchors the sequences (aligned in both, never edited in practice)
    z_0 = F.pad(z_0, (1, 0), value=tok.bos_token)
    z_1 = F.pad(z_1, (1, 0), value=tok.bos_token)

    prompt = pad_stack(prompt_list, tok.pad_token)
    prompt_pad_mask = (prompt == tok.pad_token)
    return prompt, prompt_pad_mask, z_0, z_1


def compute_batch_loss(model, prompt_list, target_list, t, coupling, tok, scheduler, device,
                       t_eps: float = 1e-2):
    """Bregman divergence loss (Eq. 23 in the Edit Flows paper) for one batch.

    Returns the scalar loss tensor plus detached per-batch edit-rate stats.
    t is clamped away from {0, 1} so κ̇/(1-κ) stays finite (linear kappa → 1/(1-t)).
    """
    t = t.clamp(min=t_eps, max=1.0 - t_eps)
    prompt, prompt_pad_mask, z_0, z_1 = make_training_batch(prompt_list, target_list, coupling, tok)

    z_t = sample_cond_pt(x2prob(z_0, tok.num_z_classes), x2prob(z_1, tok.num_z_classes), t, scheduler)
    x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, tok.gap_token, tok.pad_token)

    # Which operations bring z_t closer to z_1: (batch, z_len, 2 * vocab + 1)
    uz_mask = make_ut_mask_from_z(z_t, z_1, vocab_size=tok.vocab_size,
                                  gap_token=tok.gap_token, pad_token=tok.pad_token)

    u_t, ins_probs, sub_probs = model.forward(
        prompt=prompt.to(device),
        prompt_pad_mask=prompt_pad_mask.to(device),
        tokens=x_t.to(device),
        time_step=t.to(device),
        padding_mask=x_pad_mask.to(device),
    )
    lambda_ins = u_t[:, :, 0]
    lambda_sub = u_t[:, :, 1]
    lambda_del = u_t[:, :, 2]

    u_tia_ins = lambda_ins.unsqueeze(-1) * ins_probs    # (batch, x_len, vocab)
    u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs    # (batch, x_len, vocab)
    u_tia_del = lambda_del.unsqueeze(-1)                # (batch, x_len, 1)

    ux_cat = torch.cat([u_tia_ins, u_tia_sub, u_tia_del], dim=-1)
    uz_cat = fill_gap_tokens_with_repeats(
        ux_cat, z_gap_mask.to(device), z_pad_mask.to(device))
    u_tot = u_t.sum(dim=(1, 2))

    kappa = scheduler(t)
    kappa_dot = scheduler.derivative(t)
    sched_coeff = (kappa_dot / (1.0 - kappa).clamp(min=t_eps)).to(device)
    log_uz_cat = torch.clamp(uz_cat.log(), min=-20)
    weighted = log_uz_cat * uz_mask.to(device) * sched_coeff.unsqueeze(-1)
    loss = u_tot - weighted.sum(dim=(1, 2))
    loss = loss.mean()

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"

    with torch.no_grad():
        u_con = (uz_cat * uz_mask.to(device)).sum(dim=(1, 2)).mean()
        vocab = tok.vocab_size
        ins_loss = (lambda_ins.sum(dim=1) - weighted[:, :, :vocab].sum(dim=(1, 2))).mean()
        sub_loss = (lambda_sub.sum(dim=1) - weighted[:, :, vocab:2 * vocab].sum(dim=(1, 2))).mean()
        del_loss = (lambda_del.sum(dim=1) - weighted[:, :, 2 * vocab:].sum(dim=(1, 2))).mean()
    stats = {
        "u_tot": u_tot.mean().item(),
        "u_ins": lambda_ins.sum(dim=1).mean().item(),
        "u_del": lambda_del.sum(dim=1).mean().item(),
        "u_sub": lambda_sub.sum(dim=1).mean().item(),
        "u_con": u_con.item(),
        "ins_loss": ins_loss.item(),
        "del_loss": del_loss.item(),
        "sub_loss": sub_loss.item(),
    }
    return loss, stats


def evaluate(model, dataset: CodeGenDataset, coupling, tok, scheduler, device,
             batch_size: int, seed: int, max_examples: int | None = None,
             t_eps: float = 1e-2) -> float:
    """Mean Bregman loss over (a prefix of) the validation set.

    The RNG (time steps t, coupling and pt samples) is forked and re-seeded so
    consecutive evaluations are comparable and training randomness is untouched.
    """
    model.eval()
    n = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    total, count = 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for i in range(0, n, batch_size):
            prompt_list = dataset.prompts[i:i + batch_size]
            target_list = dataset.targets[i:i + batch_size]
            t = torch.rand(len(prompt_list), 1)
            loss, _ = compute_batch_loss(model, prompt_list, target_list, t,
                                         coupling, tok, scheduler, device,
                                         t_eps=t_eps)
            total += loss.item() * len(prompt_list)
            count += len(prompt_list)
    model.train()
    return total / max(count, 1)


def evaluate_task(model, dataset: CodeGenDataset, tok, scheduler, device,
                  num_samples: int, num_steps: int, batch_size: int, seed: int,
                  task: str = "gsm8k"):
    """Prompt-conditional generate + protocol test/ scoring."""
    if num_samples <= 0 or len(dataset) == 0:
        return {}, []
    from sample import euler_sample

    n = min(num_samples, len(dataset))
    model.eval()
    records = []
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            prompt = pad_stack(dataset.prompts[sl], tok.pad_token).to(device)
            x_final = euler_sample(
                model, tok, prompt, scheduler,
                num_steps=num_steps, max_gen_len=min(512, model.max_seq_len),
                device=device, show_progress=False)
            for j, idx in enumerate(range(sl.start, sl.stop)):
                gen = tok.decode(x_final[j])
                prompt_text, gold_text = dataset.texts[idx]
                if task in {"apps", "taco"}:
                    if score_apps_record is None:
                        raise RuntimeError("APPS/TACO eval requires score_apps_record in eval_task.py")
                    py = score_apps_record(gen, gold_text)
                    match = {
                        "correct": py.get("correct"),
                        "extracted": py.get("extracted"),
                        "extracted_answer": py.get("extracted_answer"),
                        "gold": py.get("gold", gold_text),
                    }
                else:
                    match = score_gsm8k_record(gen, gold_text)
                    py = score_gsm8k_python_record(gen, gold_text)
                records.append({
                    "prompt": prompt_text,
                    "generation": gen,
                    "gold": match["gold"],
                    "extracted_answer": match["extracted_answer"],
                    "correct": match["correct"],
                    "extracted": match["extracted"],
                    "correct_match": match["correct"],
                    "extracted_match": match["extracted"],
                    "correct_pass1": py["correct"],
                    "exec_error": py["exec_error"],
                    "length": len(gen),
                })
    metrics = aggregate_task_metrics(records, task=task)
    metrics["perf/test_s"] = time.perf_counter() - t0
    model.train()
    return metrics, records


def build_coupling(name: str, tok: ByteTokenizer, max_target_len: int) -> Coupling:
    if name == "empty":
        return EmptyCoupling()
    if name == "extended":
        return ExtendedCoupling(n_insert=32, vocab_size=tok.byte_vocab, pad_token=tok.pad_token)
    if name == "uniform":
        return UniformCoupling(min_len=1, max_len=max_target_len,
                               vocab_size=tok.byte_vocab, pad_token=tok.pad_token,
                               mirror_len=True)
    raise ValueError(f"Unknown coupling {name!r}")


def save_metrics(metrics: dict, save_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(metrics["loss"], label='Raw Loss', color='lightblue', alpha=0.6)
    plt.plot(pd.Series(metrics["loss"]).ewm(alpha=0.1).mean(), label='Smoothed (EMA)', color='blue', linewidth=2)
    if metrics.get("val_step"):
        plt.plot(metrics["val_step"], metrics["val_loss"], label='Val Loss',
                 color='red', marker='o', markersize=3, linewidth=1.5)
    plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 3, 2)
    for key, color in [("u_ins", "green"), ("u_del", "red"), ("u_sub", "purple")]:
        plt.plot(metrics[key], label=key, color=color)
    plt.xlabel('Step'); plt.title('Edit rates'); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 3, 3)
    for key, color in [("u_tot", "orange"), ("u_con", "brown")]:
        plt.plot(metrics[key], label=key, color=color)
    plt.xlabel('Step'); plt.title('Total vs constructive rate'); plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="gsm8k")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap on raw examples loaded (required for tinygsm)")
    parser.add_argument("--keep-docstring", action="store_true",
                        help="Keep the question docstring inside TinyGSM code targets")
    parser.add_argument("--max-prompt-len", type=int, default=512, help="Max prompt length in bytes")
    parser.add_argument("--max-target-len", type=int, default=512, help="Max target length in bytes")
    parser.add_argument("--coupling", choices=("empty", "extended", "uniform"), default="empty")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Micro-batches per optimizer step (effective batch = batch-size * grad-accum)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear LR warmup steps followed by cosine decay to 0 at --steps (0 = constant LR)")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scheduler-a", type=float, default=1.0)
    parser.add_argument("--scheduler-b", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results/run")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to continue from (model.pt / model_best.pt)")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2000,
                        help="Validate every N steps (0 disables validation and early stopping)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop after N validations without val_loss improvement")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum val_loss improvement to count as progress")
    parser.add_argument("--val-ratio", type=float, default=0.02,
                        help="Train fraction held out for validation on datasets without a test split")
    parser.add_argument("--val-max-examples", type=int, default=256,
                        help="Cap on val-loss examples (0 = full split)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Global grad-norm clip (0 disables)")
    parser.add_argument("--t-eps", type=float, default=1e-2,
                        help="Clamp t to [eps, 1-eps] so the Bregman coeff κ̇/(1-κ) stays finite")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16",
                        help="CUDA autocast dtype (bf16 matches the other Capella runs)")
    parser.add_argument("--task-eval-samples", type=int, default=16,
                        help="GSM8K generations to score each val (0 disables task eval)")
    parser.add_argument("--task-eval-steps", type=int, default=128,
                        help="Euler steps for in-training task eval")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Log metrics to this wandb project (default: wandb disabled)")
    parser.add_argument("--wandb-name", type=str, default=None, help="wandb run name")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, save dir: {save_dir}")

    wandb_run = None
    if args.wandb_project:
        import wandb
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = "online"
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            mode="online",
            tags=["editflow", args.dataset, "shared-params", "L256"],
        )
        run_id = os.environ.get("WANDB_RUN_ID")
        if run_id:
            wandb_kwargs["id"] = run_id
            wandb_kwargs["resume"] = "allow"
        wandb_run = wandb.init(**wandb_kwargs)

    tok = ByteTokenizer()
    pairs = load_pairs(args.dataset, split="train", max_examples=args.max_examples,
                       keep_docstring=args.keep_docstring)

    val_pairs = None
    if args.val_every > 0:
        if args.dataset in {"gsm8k", "apps", "taco"}:
            val_pairs = load_pairs(args.dataset, split="test")
        else:
            n_val = max(1, int(len(pairs) * args.val_ratio))
            split_idx = np.random.default_rng(args.seed).permutation(len(pairs))
            val_pairs = [pairs[i] for i in split_idx[:n_val]]
            pairs = [pairs[i] for i in split_idx[n_val:]]

    dataset = CodeGenDataset(pairs, tok, args.max_prompt_len, args.max_target_len)
    val_dataset = (CodeGenDataset(val_pairs, tok, args.max_prompt_len, args.max_target_len)
                   if val_pairs is not None else None)
    task_dataset = None
    task_name = args.dataset if args.dataset in {"gsm8k", "tinygsm", "apps", "taco"} else "gsm8k"
    if args.task_eval_samples > 0:
        if args.dataset in {"apps", "taco"}:
            task_pairs = load_eval_pairs(args.dataset)
            task_dataset = CodeGenDataset(
                task_pairs, tok, args.max_prompt_len, args.max_target_len,
                filter_target=False)
        else:
            gsm8k_task_pairs = load_pairs("gsm8k", split="test")
            task_dataset = CodeGenDataset(
                gsm8k_task_pairs, tok, args.max_prompt_len, args.max_target_len)

    # +1 for the BOS prefix; generation can overshoot the data length a bit
    max_seq_len = max(args.max_prompt_len, args.max_target_len + 1) + 64
    start_step = 0
    ckpt = None
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        start_step = int(ckpt.get("step", 0) or 0)
        if start_step <= 0:
            start_step = 20000
        print(f"Resuming from {args.resume} at step {start_step}")
    else:
        model = CondEditFlowsTransformer(
            vocab_size=tok.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=max_seq_len,
            bos_token_id=tok.bos_token,
            pad_token_id=tok.pad_token,
            dropout=args.dropout,
        ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.999), eps=1e-8)
    if ckpt is not None and ckpt.get("optimizer_state_dict"):
        optim.load_state_dict(ckpt["optimizer_state_dict"])
    if args.warmup_steps > 0:
        def lr_lambda(s):
            if s < args.warmup_steps:
                return (s + 1) / args.warmup_steps
            progress = (s - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    else:
        lr_sched = None
    if start_step > 0 and lr_sched is not None:
        lr_sched.last_epoch = start_step
        for param_group, lr in zip(optim.param_groups, lr_sched.get_last_lr()):
            param_group["lr"] = lr
    use_bf16 = args.precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if args.precision == "bf16" and not use_bf16:
        print("bf16 requested but not available; using fp32")
    if use_bf16:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"precision={'bf16' if use_bf16 else 'fp32'} grad_clip={args.grad_clip} "
          f"t_eps={args.t_eps} val_every={args.val_every} warmup={args.warmup_steps}")

    coupling = build_coupling(args.coupling, tok, args.max_target_len)
    scheduler = CubicScheduler(a=args.scheduler_a, b=args.scheduler_b)
    ckpt_extra = {
        "dataset": args.dataset,
        "coupling": args.coupling,
        "scheduler_a": args.scheduler_a,
        "scheduler_b": args.scheduler_b,
        "max_prompt_len": args.max_prompt_len,
        "max_target_len": args.max_target_len,
        "tokenizer": "byte",
    }
    val_cap = None if args.val_max_examples <= 0 else args.val_max_examples

    metrics = defaultdict(list)
    best_val = float("inf")
    best_task = -1.0
    bad_evals = 0
    stopped_early = False
    model.train()
    pbar = tqdm(range(start_step, args.steps), desc="Training Edit Flows", unit="step")
    for step in pbar:
        t_step = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        n_tok = 0
        step_loss = 0.0
        step_stats = defaultdict(float)
        did_backward = False
        for _ in range(args.grad_accum):
            prompt_list, target_list = dataset.sample_batch(args.batch_size, rng)
            n_tok += sum(len(p) + len(t) for p, t in zip(prompt_list, target_list))
            t = torch.rand(args.batch_size, 1)
            try:
                with autocast_ctx:
                    loss, stats = compute_batch_loss(
                        model, prompt_list, target_list, t, coupling, tok, scheduler, device,
                        t_eps=args.t_eps)
                if not torch.isfinite(loss):
                    raise ValueError("Loss is NaN or Inf")
            except (ValueError, AssertionError) as exc:
                msg = str(exc)
                if "NaN" not in msg and "Inf" not in msg and "nan" not in msg.lower():
                    raise
                print(f"skipping non-finite batch at step {step}: {exc}")
                continue
            (loss.float() / args.grad_accum).backward()
            did_backward = True
            step_loss += float(loss.detach()) / args.grad_accum
            for key, value in stats.items():
                step_stats[key] += value / args.grad_accum
        if not did_backward:
            optim.zero_grad(set_to_none=True)
            continue
        grad_norm = 0.0
        if args.grad_clip and args.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(grad_norm):
            print(f"skipping non-finite grad at step {step}: {grad_norm}")
            optim.zero_grad(set_to_none=True)
            continue
        optim.step()
        if lr_sched is not None:
            lr_sched.step()
        step_s = time.perf_counter() - t_step

        metrics["loss"].append(step_loss)
        for key, value in step_stats.items():
            metrics[key].append(value)
        if wandb_run is not None:
            payload = {
                "train/loss": step_loss,
                "train/ins_loss": step_stats["ins_loss"],
                "train/del_loss": step_stats["del_loss"],
                "train/sub_loss": step_stats["sub_loss"],
                "train/lr": optim.param_groups[0]["lr"],
                "healthy/grad_norm": grad_norm,
                "stats/u_ins": step_stats["u_ins"],
                "stats/u_del": step_stats["u_del"],
                "stats/u_sub": step_stats["u_sub"],
                "stats/u_tot": step_stats["u_tot"],
                "stats/u_con": step_stats["u_con"],
                "perf/train_step_s": step_s,
            }
            if step_s > 0 and n_tok > 0:
                payload["perf/train_tokens_per_s"] = n_tok / step_s
            wandb_run.log(payload, step=step + 1)

        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{step_loss:.1f}",
                             u_ins=f"{metrics['u_ins'][-1]:.1f}",
                             u_del=f"{metrics['u_del'][-1]:.1f}",
                             u_sub=f"{metrics['u_sub'][-1]:.1f}")

        if val_dataset is not None and (step + 1) % args.val_every == 0:
            t_val = time.perf_counter()
            val_loss = evaluate(
                model, val_dataset, coupling, tok, scheduler, device,
                args.batch_size, seed=args.seed + 1, max_examples=val_cap,
                t_eps=args.t_eps)
            val_s = time.perf_counter() - t_val
            metrics["val_loss"].append(val_loss)
            metrics["val_step"].append(step + 1)
            val_payload = {
                "val/nll": val_loss,
                "perf/val_step_s": val_s,
            }
            task_metrics, task_records = {}, []
            if args.task_eval_samples > 0 and task_dataset is not None:
                task_metrics, task_records = evaluate_task(
                    model, task_dataset, tok, scheduler, device,
                    num_samples=args.task_eval_samples,
                    num_steps=args.task_eval_steps,
                    batch_size=min(args.batch_size, 8),
                    seed=args.seed + 2,
                    task=task_name)
                val_payload.update(task_metrics)
            if wandb_run is not None:
                wandb_run.log(val_payload, step=step + 1)
                wandb_run.summary["best_val_nll"] = min(best_val, val_loss) if math.isfinite(best_val) else val_loss
                if "test/pass@1_match" in task_metrics:
                    prev = wandb_run.summary.get("best_test_pass@1_match")
                    if prev is None or task_metrics["test/pass@1_match"] > float(prev):
                        wandb_run.summary["best_test_pass@1_match"] = task_metrics["test/pass@1_match"]
                if "test/pass@1" in task_metrics:
                    prev = wandb_run.summary.get("best_test_pass@1")
                    if prev is None or task_metrics["test/pass@1"] > float(prev):
                        wandb_run.summary["best_test_pass@1"] = task_metrics["test/pass@1"]
                # if task_records:
                #     wandb_run.log({
                #         "val/samples": wandb.Table(
                #             columns=["prompt", "generation", "gold", "extracted_answer", "correct", "length"],
                #             data=[[
                #                 rec.get("prompt", "")[:500],
                #                 rec.get("generation", "")[:500],
                #                 rec.get("gold"),
                #                 rec.get("extracted_answer"),
                #                 rec.get("correct"),
                #                 rec.get("length"),
                #             ] for rec in task_records[:16]],
                #         ),
                #     }, step=step + 1)
            em = task_metrics.get("test/pass@1_match")
            if em is not None and em > best_task:
                best_task = em
                save_checkpoint(
                    save_dir / "model_best_task.pt", model, optim,
                    extra={**ckpt_extra, "step": step + 1, "val_loss": val_loss,
                           "val_pass@1_match": em})
            if val_loss < best_val - args.min_delta:
                best_val = val_loss
                bad_evals = 0
                save_checkpoint(save_dir / "model_best.pt", model, optim,
                                extra={**ckpt_extra, "step": step + 1, "val_loss": val_loss})
                pbar.write(f"step {step + 1}: val/nll={val_loss:.2f} (new best"
                           + (f", test/pass@1_match={em:.3f}" if em is not None else "")
                           + ")")
            else:
                bad_evals += 1
                pbar.write(f"step {step + 1}: val/nll={val_loss:.2f} "
                           f"(best {best_val:.2f}, no improvement {bad_evals}/{args.patience}"
                           + (f", test/pass@1_match={em:.3f}" if em is not None else "")
                           + ")")
            past_warmup = (step + 1) > max(args.warmup_steps, args.val_every)
            if past_warmup and bad_evals >= args.patience:
                pbar.write(f"Early stopping at step {step + 1}: "
                           f"no val/nll improvement in {args.patience} validations")
                stopped_early = True

        if (step + 1) % args.save_every == 0 or step == args.steps - 1 or stopped_early:
            save_checkpoint(save_dir / "model.pt", model, optim,
                            extra={**ckpt_extra, "step": step + 1})
            save_metrics(metrics, save_dir)

        if stopped_early:
            break

    print(f"Done. Checkpoint and metrics saved to {save_dir}/")
    if val_dataset is not None:
        print(f"Best val/nll: {best_val:.2f} (checkpoint: {save_dir}/model_best.pt)")
        if best_task >= 0:
            print(f"Best test/pass@1_match: {best_task:.3f} (checkpoint: {save_dir}/model_best_task.pt)")
    if wandb_run is not None:
        if val_dataset is not None:
            wandb_run.summary["best_val_nll"] = best_val
            if best_task >= 0:
                wandb_run.summary["best_test_pass@1_match"] = best_task
        wandb_run.finish()


if __name__ == "__main__":
    main()
