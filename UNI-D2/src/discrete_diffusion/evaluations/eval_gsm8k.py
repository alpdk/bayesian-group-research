"""GSM8K prefix eval from a UNI-D2 Lightning checkpoint.

Example:
  PYTHONPATH=src python -m discrete_diffusion.evaluations.eval_gsm8k \\
      --checkpoint path/to/best.ckpt --out eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from discrete_diffusion.data import get_tokenizer
from discrete_diffusion.evaluations.protocol import save_protocol_artifacts
from discrete_diffusion.evaluations.task import (
    aggregate_task_metrics,
    encode_question_prefixes,
    load_task_split,
    pass_at_1_k_values,
    score_gsm8k_python_record,
    score_gsm8k_record,
    score_tinygsm_record,
)


def _load_model(checkpoint_path: str, device: torch.device):
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  if "hyper_parameters" not in ckpt or "config" not in ckpt["hyper_parameters"]:
    raise ValueError(f"Checkpoint missing hyper_parameters.config: {checkpoint_path}")
  model_config = ckpt["hyper_parameters"]["config"]
  if not isinstance(model_config, DictConfig):
    model_config = OmegaConf.create(model_config)
  tokenizer = get_tokenizer(model_config)
  algo_cls = hydra.utils.get_class(model_config.algo._target_)
  model = algo_cls.load_from_checkpoint(
      checkpoint_path, config=model_config, tokenizer=tokenizer, map_location=device)
  model.to(device)
  model.eval()
  if hasattr(model, "_eval_mode"):
    model._eval_mode()
  return model, model_config, tokenizer


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", required=True, help="Output JSON path")
  parser.add_argument("--num-samples", type=int, default=0,
                      help="Cap on eval split (0 = full split)")
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--num-steps", type=int, default=128)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--task", type=str,
                      choices=("gsm8k", "gsm8k-python", "tinygsm"),
                      default="gsm8k")
  parser.add_argument("--tinygsm-cache", type=str, default=None)
  parser.add_argument("--wandb-project", type=str, default=None)
  parser.add_argument("--wandb-name", type=str, default=None)
  args = parser.parse_args()

  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  torch.set_float32_matmul_precision("high")
  torch.set_grad_enabled(False)

  ckpt_path = str(Path(args.checkpoint).resolve())
  print(f"Loading {ckpt_path}", flush=True)
  model, config, tokenizer = _load_model(ckpt_path, device)

  questions, golds = load_task_split(
      args.task, args.num_samples, tinygsm_cache=args.tinygsm_cache)
  n = len(questions)
  print(
      f"{args.task} n={n} batch={args.batch_size} steps={args.num_steps}",
      flush=True)

  records = []
  all_tokens = []
  metrics = {}
  ks = pass_at_1_k_values(config)
  t0 = time.perf_counter()
  for k in ks:
    k_records = []
    k_tokens = []
    for start in range(0, n, args.batch_size):
      qs = questions[start:start + args.batch_size]
      gs = golds[start:start + args.batch_size]
      prefixes = encode_question_prefixes(tokenizer, qs)
      samples = model.generate_samples(
          num_samples=len(prefixes), num_steps=args.num_steps,
          prefix=prefixes, unmask_k=k)
      samples = samples.detach().cpu()
      k_tokens.append(samples)
      decoded = tokenizer.batch_decode(samples.tolist(), skip_special_tokens=True)
      pad_id = tokenizer.pad_token_id
      for i, gen in enumerate(decoded):
        gold = gs[i]
        match = score_gsm8k_record(gen, gold)
        if args.task == "tinygsm":
          py = score_tinygsm_record(gen, gold)
        else:
          py = score_gsm8k_python_record(gen, gold)
        primary = match if args.task == "gsm8k" else py
        length = int((samples[i] != pad_id).sum().item()) if pad_id is not None else int(samples[i].numel())
        k_records.append({
            "prompt": qs[i],
            "generation": gen,
            "gold": primary.get("gold", gold),
            "gold_answer": gold,
            "extracted_answer": primary.get("extracted_answer"),
            "correct": primary.get("correct"),
            "extracted": primary.get("extracted"),
            "correct_pass1": py.get("correct"),
            "exec_error": py.get("exec_error"),
            "length": length,
            "prompt_len": int(prefixes[i].numel()),
        })
      done = min(start + args.batch_size, n)
      acc = sum(r["correct_pass1"] for r in k_records) / len(k_records)
      print(f"  k={k} {done}/{n}  pass@1={acc:.4f}", flush=True)
    metrics.update(aggregate_task_metrics(k_records, args.task, k=k))
    if int(k) == 1 or not records:
      records = k_records
      all_tokens = k_tokens
  metrics["perf/test_s"] = time.perf_counter() - t0
  tokens = torch.cat(all_tokens, dim=0)
  out_path = Path(args.out)
  save_protocol_artifacts(out_path.parent, 0, tokens, records, metrics)
  payload = {
      "checkpoint": ckpt_path,
      "task": args.task,
      "n": n,
      "num_steps": args.num_steps,
      "metrics": metrics,
      "records": records,
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(json.dumps({"out": str(out_path), "n": n, **metrics}, indent=2), flush=True)

  if args.wandb_project:
    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        job_type="eval",
        config={"checkpoint": ckpt_path, "n": n, "num_steps": args.num_steps, "task": args.task},
        tags=["shared-params", f"{args.task}-eval", "L256"],
    )
    run.log(metrics, step=0)
    run.summary["best_test_pass@1_k1"] = metrics.get("test/pass@1_k1")
    run.summary.update(metrics)
    run.finish()


if __name__ == "__main__":
  main()
