"""Standalone GSM8K evaluation of a saved PUMA checkpoint.

Usage (single GPU):
    python evaluate.py --ckpt ckpts/<date>/best/ema_step=235000.pt

Usage (multi-GPU, faster):
    torchrun --standalone --nproc_per_node=4 evaluate.py --ckpt <path> \
        --confidence top_k top_k_margin --unmasking_num 1 2 3
"""
import argparse
import math

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from model.transformer import MDMTransformer, MDMConfig
from sampling import mdm_sampling
from eval.gsm8k_eval import (
    test_gsm8k_tokenization, get_tokenizer,
    _extract_code, _safe_exec_no_timer, _time_limit, _Timeout,
    _to_number, _numbers_equal,
)
from train import setup_ddp


def score_sample(sample: str, answer: str, timeout_s: float = 1.0) -> dict:
    """Per-sample breakdown of where generation succeeds or fails."""
    m = {"code_defined": 0, "exec_ok": 0, "numeric_out": 0, "correct": 0}
    code = _extract_code(sample)
    try:
        with _time_limit(timeout_s):
            ns = _safe_exec_no_timer(code)
            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return m
            m["code_defined"] = 1
            out = fn()
    except (_Timeout, Exception):
        return m
    m["exec_ok"] = 1
    pred = _to_number(out)
    if pred is None:
        return m
    m["numeric_out"] = 1
    if _numbers_equal(pred, _to_number(answer)):
        m["correct"] = 1
    return m


@torch.no_grad()
def evaluate_gsm8k_metrics(model, cfg, device, rank, world_size, sampling):
    """Like eval.gsm8k_eval.evaluate_ddp_gsm8k, but returns a metric breakdown
    instead of a single accuracy number."""
    mask_id = cfg.data.mask_id
    max_len = int(cfg.model.max_position)
    X, answers = test_gsm8k_tokenization(mask_id, max_len=max_len)
    N_val = len(X)

    per_rank = math.ceil(N_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, N_val)
    batch_size = 32
    num_batches = math.ceil((end - start) / batch_size)

    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id
    keys = ["code_defined", "exec_ok", "numeric_out", "correct"]
    counters = {k: 0 for k in keys}
    total, gen_len_sum = 0, 0

    for j in tqdm(range(num_batches), desc="Evaluating", disable=(rank != 0)):
        s = start + j * batch_size
        e = min(s + batch_size, end)
        batch_X = torch.from_numpy(X[s:e]).long().to(device)
        samples_tensor = mdm_sampling(model, batch_X, mask_id, sampling, device,
                                      arm_init=cfg.model.arm_init != "none")
        samples_tensor = samples_tensor.masked_fill(samples_tensor == mask_id, pad_id)

        # generated answer length = non-pad tokens in output minus prompt tokens
        prompt_lens = (batch_X != mask_id).sum(dim=1)
        out_lens = (samples_tensor != pad_id).sum(dim=1)
        gen_len_sum += (out_lens - prompt_lens).clamp(min=0).sum().item()

        samples = tokenizer.batch_decode(samples_tensor.cpu().numpy(), skip_special_tokens=True)
        for sample, answer in zip(samples, answers[s:e]):
            m = score_sample(sample, answer)
            for k in keys:
                counters[k] += m[k]
            total += 1

    t = torch.tensor([counters[k] for k in keys] + [total, gen_len_sum],
                     dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vals = t.tolist()
    g_total = vals[len(keys)]
    return {
        "accuracy": vals[keys.index("correct")] / g_total,
        "code_defined_rate": vals[keys.index("code_defined")] / g_total,
        "exec_ok_rate": vals[keys.index("exec_ok")] / g_total,
        "numeric_out_rate": vals[keys.index("numeric_out")] / g_total,
        "wrong_answer_rate": (vals[keys.index("numeric_out")] - vals[keys.index("correct")]) / g_total,
        "avg_gen_tokens": vals[len(keys) + 1] / g_total,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--confidence", nargs="+", default=["top_k"],
                    help="one or more of: top_k, top_k_margin, entropy")
    ap.add_argument("--unmasking_num", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--temperature", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    model = MDMTransformer(MDMConfig(**cfg.model)).to(device)
    # EMA snapshots already store EMA weights in model_state_dict
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if is_main:
        print(f"Checkpoint: {args.ckpt}")
        print(f"  step={ckpt.get('global_step')} val_loss={ckpt.get('val_loss')} "
              f"ema={ckpt.get('is_ema_snapshot')} max_position={cfg.model.max_position}")
        print(f"Evaluating GSM8K test set on {world_size} GPU(s)")

    results = {}
    with torch.inference_mode():
        for conf in args.confidence:
            for k in args.unmasking_num:
                sampling = OmegaConf.create({
                    "temperature": args.temperature,
                    "confidence": conf,
                    "unmasking_num": int(k),
                    "eos_id": int(cfg.training.eos_id),
                })
                metrics = evaluate_gsm8k_metrics(model, cfg, device, rank, world_size, sampling)
                results[f"{conf}_unmasking_{k}"] = metrics
                if is_main:
                    metric_str = " ".join(f"{k_}={v:.4f}" for k_, v in metrics.items())
                    print(f"RESULT confidence={conf} unmasking_num={k} "
                          f"temperature={args.temperature} {metric_str}", flush=True)

    if is_main:
        print("=== summary (sorted by accuracy) ===")
        for name, m in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]):
            print(f"{name}: acc={m['accuracy']:.4f} code_defined={m['code_defined_rate']:.4f} "
                  f"exec_ok={m['exec_ok_rate']:.4f} numeric={m['numeric_out_rate']:.4f} "
                  f"wrong_answer={m['wrong_answer_rate']:.4f} avg_gen_tokens={m['avg_gen_tokens']:.1f}")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
