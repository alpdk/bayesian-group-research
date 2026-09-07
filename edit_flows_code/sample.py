"""Sample solutions from a trained conditional Edit Flows model.

Prompts are taken from the GSM8K test split (also the standard eval for
TinyGSM-trained models). For GSM8K-style generations the final "#### <num>"
answer is compared to gold; for TinyGSM-style Python generations pass
--execute to run simple_math_problem() in a subprocess and compare results.

Example:
    python sample.py --checkpoint results/run/model.pt --num-prompts 8 --num-steps 500
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from tqdm import tqdm

from eval_task import (
    aggregate_task_metrics,
    extract_gsm8k_answer,
    score_apps_record,
    score_gsm8k_python_record,
    score_gsm8k_record,
    score_tinygsm_record,
)
from flow import CubicScheduler, KappaScheduler
from model import load_checkpoint
from tokenizer import ByteTokenizer


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


def apply_ins_del_operations(
    x_t: torch.Tensor,
    ins_mask: torch.Tensor,
    del_mask: torch.Tensor,
    ins_tokens: torch.Tensor,
    pad_token: int,
    max_seq_len: int = 1024,
) -> torch.Tensor:
    """Applies insertion and deletion operations to x_t based on the masks."""
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substitutions
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]

    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    xt_pad_mask = (x_t == pad_token)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        print(f"Unexpected max_new_len <= 0: {max_new_len}, did we delete everything?")
        return torch.full((batch_size, 1), pad_token, dtype=x_t.dtype, device=device)

    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device)

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)
    cum_ins_before = torch.nn.functional.pad(cum_ins[:, :-1], (1, 0), value=0)

    # Place non-deleted tokens at their shifted positions
    new_pos = pos_idx + cum_ins_before - cum_del
    keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]

    # Place insertions one slot after the shifted position
    if eff_ins_mask.any():
        ins_pos = new_pos + 1
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)
        if ins_valid.any():
            x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]

    if max_new_len > max_seq_len:
        print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
        max_new_len = max_seq_len

    return x_new[:, :max_new_len]


def get_adaptive_h(h: float, t: torch.Tensor, scheduler: KappaScheduler):
    deriv = scheduler.derivative(t).clamp_min(1e-6)
    coeff = (1 - scheduler(t)).clamp_min(0) / deriv
    _h = h * torch.ones_like(t, device=t.device)
    return torch.minimum(_h, coeff)


@torch.no_grad()
def euler_sample(
    model,
    tok: ByteTokenizer,
    prompt: torch.Tensor,             # (batch, p_len)
    scheduler: KappaScheduler,
    num_steps: int = 500,
    max_gen_len: int = 512,
    device: torch.device = torch.device("cpu"),
    show_progress: bool = True,
):
    """Transports x_0 = [BOS] to a sample of the target distribution via Euler steps."""
    model.eval()
    batch_size = prompt.shape[0]
    prompt = prompt.to(device)
    prompt_pad_mask = (prompt == tok.pad_token)

    default_h = 1 / num_steps
    t = torch.zeros(batch_size, 1, device=device)
    x_t = torch.full((batch_size, 1), tok.bos_token, dtype=torch.long, device=device)

    pbar = tqdm(desc="Euler sampling", disable=not show_progress)
    n_iter = 0
    max_iters = max(num_steps * 4, num_steps + 1)
    while float(t.max()) <= 1 - default_h and n_iter < max_iters:
        n_iter += 1
        x_pad_mask = (x_t == tok.pad_token)
        u_t, ins_probs, sub_probs = model.forward(
            prompt=prompt,
            prompt_pad_mask=prompt_pad_mask,
            tokens=x_t,
            time_step=t,
            padding_mask=x_pad_mask,
        )
        lambda_ins = u_t[:, :, 0].clamp(max=50)
        lambda_sub = u_t[:, :, 1].clamp(max=50)
        lambda_del = u_t[:, :, 2].clamp(max=50)

        adapt_h = get_adaptive_h(default_h, t, scheduler)

        ins_mask = torch.rand(lambda_ins.shape, device=device) < (
            1 - torch.exp(-adapt_h * lambda_ins))
        del_sub_mask = torch.rand(lambda_sub.shape, device=device) < (
            1 - torch.exp(-adapt_h * (lambda_sub + lambda_del)))

        prob_del = torch.where(
            del_sub_mask,
            lambda_del / (lambda_sub + lambda_del + 1e-12),
            torch.zeros_like(lambda_del))
        del_mask = torch.bernoulli(prob_del.clamp(0, 1)).bool()
        sub_mask = del_sub_mask & ~del_mask

        # Keep the BOS anchor intact and cap the generation length
        del_mask[:, 0] = False
        sub_mask[:, 0] = False
        if x_t.shape[1] >= max_gen_len:
            ins_mask[:] = False

        ins_tokens = torch.full(ins_probs.shape[:2], tok.pad_token, dtype=torch.long, device=device)
        sub_tokens = torch.full(sub_probs.shape[:2], tok.pad_token, dtype=torch.long, device=device)
        non_pad_mask = ~x_pad_mask
        if non_pad_mask.any():
            ins_tokens[non_pad_mask] = torch.multinomial(
                ins_probs[non_pad_mask].clamp_min(1e-8), num_samples=1).squeeze(-1)
            sub_tokens[non_pad_mask] = torch.multinomial(
                sub_probs[non_pad_mask].clamp_min(1e-8), num_samples=1).squeeze(-1)

        x_t[sub_mask] = sub_tokens[sub_mask]
        x_t = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            pad_token=tok.pad_token, max_seq_len=min(model.max_seq_len, max_gen_len))
        t = (t + adapt_h).clamp(max=1.0)
        pbar.update(1)
    pbar.close()
    return x_t


def execute_tinygsm(code: str, timeout: float = 5.0) -> str | None:
    """Runs generated simple_math_problem() code in a subprocess, returns printed result."""
    harness = code + "\n\nprint(simple_math_problem())\n"
    try:
        proc = subprocess.run(
            ["python", "-c", harness], capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=8,
                        help="Prompts to score (0 = full split)")
    parser.add_argument("--dataset", type=str, choices=("gsm8k", "tinygsm", "apps", "taco"), default="gsm8k")
    parser.add_argument("--tinygsm-cache", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--max-gen-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--execute", action="store_true",
                        help="Execute generated Python (TinyGSM style) to score correctness")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: <checkpoint dir>/samples.json)")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None,
                        help="Resume this W&B run id (logs eval onto an existing train run)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    tok = ByteTokenizer()
    scheduler = CubicScheduler(a=ckpt.get("scheduler_a", 1.0), b=ckpt.get("scheduler_b", 1.0))
    print(f"Loaded {args.checkpoint} (dataset={ckpt.get('dataset')}, coupling={ckpt.get('coupling')})")

    if args.dataset in {"apps", "taco"}:
        from data import load_eval_pairs
        pairs = load_eval_pairs(args.dataset)
        n = len(pairs) if args.num_prompts <= 0 else min(args.num_prompts, len(pairs))
        questions = [p[0] for p in pairs[:n]]
        gold_texts = [p[1] for p in pairs[:n]]
        score_fn = score_apps_record
        task_name = args.dataset
    elif args.dataset == "tinygsm":
        from datasets import load_from_disk
        import os
        cache = args.tinygsm_cache or os.environ.get(
            "TINYGSM_RAW",
            "/data/horse/ws/alpo658h-insert-diff/scratch/tinygsm/tinygsm_raw_n100000_seed42.dat",
        )
        split = load_from_disk(cache)["test"]
        n = len(split) if args.num_prompts <= 0 else min(args.num_prompts, len(split))
        split = split.select(range(n))
        questions = [q.strip() for q in split["question"]]
        gold_texts = [a.strip() for a in split["answer"]]
        score_fn = score_tinygsm_record
        task_name = "tinygsm"
    else:
        from datasets import load_dataset
        test = load_dataset("openai/gsm8k", "main", split="test")
        n = len(test) if args.num_prompts <= 0 else min(args.num_prompts, len(test))
        test = test.select(range(n))
        questions = [q.strip() for q in test["question"]]
        gold_texts = [a.strip() for a in test["answer"]]
        score_fn = score_gsm8k_record
        task_name = "gsm8k"
    max_prompt_len = min(int(ckpt.get("max_prompt_len", model.max_seq_len)), model.max_seq_len)
    n_trunc = sum(len(tok.encode(q)) > max_prompt_len for q in questions)
    if n_trunc:
        print(f"Truncating {n_trunc}/{len(questions)} prompts to {max_prompt_len} bytes (model max_seq_len={model.max_seq_len})")

    from data import pad_stack
    records = []
    verbose = len(questions) <= 32
    for start in range(0, len(questions), args.batch_size):
        batch_q = questions[start:start + args.batch_size]
        prompt = pad_stack([tok.encode(q)[:max_prompt_len] for q in batch_q], tok.pad_token)
        x_final = euler_sample(
            model, tok, prompt, scheduler,
            num_steps=args.num_steps, max_gen_len=args.max_gen_len, device=device)
        for i, q in enumerate(batch_q):
            gen = tok.decode(x_final[i])
            gold = gold_texts[start + i]
            match = score_gsm8k_record(gen, gold)
            if task_name == "tinygsm":
                py = score_tinygsm_record(gen, gold)
            elif task_name in {"apps", "taco"}:
                py = score_apps_record(gen, gold)
                match = {
                    "correct": py.get("correct"),
                    "extracted": py.get("extracted"),
                    "extracted_answer": py.get("extracted_answer"),
                    "gold": py.get("gold", gold),
                }
            else:
                py = score_gsm8k_python_record(gen, gold)
            primary = match if task_name == "gsm8k" else py
            record = {
                "question": q,
                "generation": gen,
                "gold_answer": gold,
                "extracted_answer": primary["extracted_answer"],
                "correct": primary["correct"],
                "extracted": primary["extracted"],
                "correct_match": match["correct"],
                "extracted_match": match["extracted"],
                "correct_pass1": py["correct"],
                "exec_error": py.get("exec_error"),
            }
            if args.execute:
                record["executed_answer"] = execute_tinygsm(gen)
            records.append(record)
            if verbose:
                print("=" * 80)
                print(f"Q: {q}")
                print(f"--- generation ---\n{gen}")
                print(f"--- gold: {record.get('extracted_answer')}, correct: {record['correct']}")
        done = min(start + args.batch_size, len(questions))
        acc = sum(r["correct"] for r in records) / len(records)
        print(f"  {done}/{len(questions)}  acc={acc:.4f}", flush=True)

    metrics = aggregate_task_metrics(records, task_name)
    if metrics:
        n_ok = int(round(metrics["test/pass@1_match"] * len(records)))
        print(f"\nAccuracy: {metrics['test/pass@1_match']:.2%} ({n_ok}/{len(records)})")
        print(f"extracted_frac: {metrics['test/answer_extracted_frac']:.2%}")
        if "test/pass@1" in metrics:
            print(f"pass@1: {metrics['test/pass@1']:.2%}  exec_error_frac: {metrics['test/exec_error_frac']:.2%}")

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "samples.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} samples to {out_path}")

    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            id=args.wandb_id,
            resume="allow" if args.wandb_id else None,
            job_type="eval",
            config={
                "checkpoint": args.checkpoint,
                "n": len(records),
                "num_steps": args.num_steps,
                "max_gen_len": args.max_gen_len,
                "max_prompt_len": max_prompt_len,
                "n_truncated_prompts": n_trunc,
                "task": task_name,
            },
            tags=["editflow", "shared-params", f"{task_name}-eval", "L256"],
        )
        run.log(metrics, step=0)
        if "test/pass@1_match" in metrics:
            run.summary["best_test_pass@1_match"] = metrics["test/pass@1_match"]
        if "test/pass@1" in metrics:
            run.summary["best_test_pass@1"] = metrics["test/pass@1"]
        run.summary.update(metrics)
        url = run.get_url()
        run.finish()
        print(f"Logged to wandb {url}")


if __name__ == "__main__":
    main()
