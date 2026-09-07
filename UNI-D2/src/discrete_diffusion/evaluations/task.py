"""GSM8K / TinyGSM task metrics from docs/research/metrics.md.

Pure-Python scoring. No model or GPU required.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence

GSM8K_ANSWER_RE = re.compile(r"####\s*([-\d.,]+)")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def normalize_number(text: Optional[str]) -> Optional[str]:
  """Strip thousands separators and canonicalise integers."""
  if text is None:
    return None
  cleaned = text.replace(",", "").strip().rstrip(".")
  if not cleaned:
    return None
  try:
    value = float(cleaned)
  except ValueError:
    return cleaned
  if value.is_integer():
    return str(int(value))
  return str(value)


def extract_gsm8k_answer(text: str) -> Optional[str]:
  """Parse the last ``#### <num>`` marker (GSM8K gold format)."""
  matches = GSM8K_ANSWER_RE.findall(text or "")
  if not matches:
    return None
  return normalize_number(matches[-1])


def extract_printed_number(text: str) -> Optional[str]:
  """Fallback: last number-like token in text."""
  matches = _NUMBER_RE.findall(text or "")
  if not matches:
    return None
  return normalize_number(matches[-1])


def _code_to_run(code: str) -> str:
  """Drop echoed prompts; keep from the first ``def`` if present."""
  text = code or ""
  idx = text.find("def simple_math_problem")
  if idx < 0:
    idx = text.find("def ")
  return text[idx:] if idx >= 0 else text


def execute_tinygsm(code: str, timeout: float = 5.0) -> Optional[str]:
  """Run ``simple_math_problem()`` in a subprocess; return printed stdout."""
  harness = _code_to_run(code) + "\n\nprint(simple_math_problem())\n"
  try:
    proc = subprocess.run(
      ["python", "-c", harness],
      capture_output=True,
      text=True,
      timeout=timeout,
    )
  except (subprocess.TimeoutExpired, OSError):
    return None
  if proc.returncode != 0:
    return None
  out = proc.stdout.strip()
  return out if out else None


def score_gsm8k_record(generation: str, gold: str) -> Dict[str, Any]:
  extracted = extract_gsm8k_answer(generation)
  gold_n = extract_gsm8k_answer(gold) or normalize_number(gold)
  extracted_ok = extracted is not None
  correct = bool(extracted_ok and gold_n is not None and extracted == gold_n)
  return {
    "extracted_answer": extracted,
    "gold": gold_n,
    "correct": correct,
    "extracted": extracted_ok,
  }


def score_gsm8k_python_record(
  generation: str,
  gold: str,
  timeout: float = 5.0,
) -> Dict[str, Any]:
  """PUMA protocol: execute generated Python, compare to GSM8K ``####`` gold."""
  executed = execute_tinygsm(generation, timeout=timeout)
  gold_n = extract_gsm8k_answer(gold) or normalize_number(gold)
  exec_ok = executed is not None
  correct = bool(
    exec_ok and gold_n is not None
    and normalize_number(executed) == gold_n
  )
  return {
    "extracted_answer": executed,
    "gold": gold_n,
    "correct": correct,
    "extracted": exec_ok,
    "exec_error": not exec_ok,
  }


def score_tinygsm_record(
  generation: str,
  gold: str,
  timeout: float = 5.0,
) -> Dict[str, Any]:
  executed = execute_tinygsm(generation, timeout=timeout)
  gold_value = execute_tinygsm(gold, timeout=timeout)
  if gold_value is None:
    gold_value = extract_gsm8k_answer(gold) or normalize_number(gold)
  exec_ok = executed is not None
  correct = bool(
    exec_ok and gold_value is not None
    and normalize_number(executed) == normalize_number(gold_value)
  )
  return {
    "extracted_answer": executed,
    "gold": gold_value,
    "correct": correct,
    "extracted": exec_ok,
    "exec_error": not exec_ok,
  }


def namespace_test_metrics(metrics: Dict[str, float], namespace: Optional[str]) -> Dict[str, float]:
  """Move ``test/foo`` → ``test/{namespace}/foo`` for a secondary eval task."""
  if not namespace:
    return metrics
  out = {}
  for key, value in metrics.items():
    if key.startswith("test/"):
      out[f"test/{namespace}/{key[len('test/'):]}"] = value
    else:
      out[key] = value
  return out


def aggregate_task_metrics(
  records: Sequence[Dict[str, Any]],
  task: str,
  key_namespace: Optional[str] = None,
) -> Dict[str, float]:
  """Return protocol ``test/`` task scalars from scored records."""
  n = len(records)
  if n == 0:
    return {}
  match_ok = sum(1 for r in records if r.get("correct_match", r.get("correct")))
  match_ext = sum(1 for r in records if r.get("extracted_match", r.get("extracted")))
  metrics = {
    "test/pass@1_match": match_ok / n,
    "test/answer_extracted_frac": match_ext / n,
  }
  has_python = (
    any("correct_pass1" in r or r.get("exec_error") is not None for r in records)
    or task in {"tinygsm", "gsm8k-python"})
  if has_python:
    if any("correct_pass1" in r for r in records):
      pass1_ok = sum(1 for r in records if r.get("correct_pass1"))
    else:
      pass1_ok = sum(1 for r in records if r.get("correct"))
    errors = sum(1 for r in records if r.get("exec_error"))
    metrics["test/pass@1"] = pass1_ok / n
    metrics["test/exec_error_frac"] = errors / n
  return namespace_test_metrics(metrics, key_namespace)


def tinygsm_raw_cache_path(
  cache_dir: str,
  max_examples: int = 100000,
  seed: int = 42,
) -> str:
  import os
  return os.path.join(
    cache_dir, f"tinygsm_raw_n{int(max_examples)}_seed{int(seed)}.dat")


def load_task_split(
  task: str,
  num_samples: int,
  *,
  config: Any = None,
  tinygsm_cache: Optional[str] = None,
) -> tuple[List[str], List[str]]:
  """Questions and gold strings for GSM8K / TinyGSM test."""
  import os
  from pathlib import Path

  from datasets import load_dataset, load_from_disk

  if task == "tinygsm":
    cache = tinygsm_cache
    if cache is None and config is not None:
      cache_dir = str(getattr(config.data, "cache_dir", "") or "")
      max_ex = int(getattr(config.data, "max_examples", 100000) or 100000)
      seed = int(getattr(config.data, "split_seed", getattr(config, "seed", 42)) or 42)
      cache = tinygsm_raw_cache_path(cache_dir, max_ex, seed)
    if cache is None:
      cache = str(
        Path(os.environ.get(
          "DISCRETE_DIFFUSION_SCRATCH_DIR", "./scratch")) / "tinygsm" /
        "tinygsm_raw_n100000_seed42.dat")
    if not os.path.exists(cache):
      from discrete_diffusion.data.loaders import _load_tinygsm_split
      cache_dir = os.path.dirname(cache)
      max_ex = 100000
      seed = 42
      if config is not None:
        cache_dir = str(getattr(config.data, "cache_dir", cache_dir) or cache_dir)
        max_ex = int(getattr(config.data, "max_examples", max_ex) or max_ex)
        seed = int(getattr(config.data, "split_seed", seed) or seed)
      _load_tinygsm_split(cache_dir, max_ex, val_ratio=0.02, seed=seed)
    ds = load_from_disk(cache)["test"]
    n = len(ds) if num_samples <= 0 else min(num_samples, len(ds))
    ds = ds.select(range(n))
    return [q.strip() for q in ds["question"]], [a.strip() for a in ds["answer"]]

  split = load_dataset("openai/gsm8k", "main", split="test")
  n = len(split) if num_samples <= 0 else min(num_samples, len(split))
  split = split.select(range(n))
  return [q.strip() for q in split["question"]], [a.strip() for a in split["answer"]]


def encode_question_prefixes(tokenizer: Any, questions: Sequence[str]) -> List[Any]:
  import torch
  newline_ids = tokenizer.encode("\n", add_special_tokens=False)
  out = []
  for q in questions:
    q_ids = tokenizer.encode(q, add_special_tokens=False)
    out.append(torch.tensor(q_ids + newline_ids, dtype=torch.long))
  return out


def infer_task_name(data_name: Optional[str]) -> Optional[str]:
  if not data_name:
    return None
  name = str(data_name).lower()
  if "tinygsm" in name or "tiny_gsm" in name:
    return "tinygsm"
  if "gsm8k" in name:
    return "gsm8k"
  return None


def wandb_table_rows(
  records: Iterable[Dict[str, Any]],
  max_rows: int = 16,
) -> List[List[Any]]:
  rows = []
  for rec in records:
    if len(rows) >= max_rows:
      break
    rows.append([
      rec.get("prompt", ""),
      rec.get("generation", ""),
      rec.get("gold", rec.get("gold_answer", "")),
      rec.get("extracted_answer"),
      rec.get("correct"),
      rec.get("length"),
    ])
  return rows


WANDB_TABLE_COLUMNS = [
  "prompt",
  "generation",
  "gold",
  "extracted_answer",
  "correct",
  "length",
]
