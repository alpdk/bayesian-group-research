"""Shared val/test logging helpers from docs/research/metrics.md."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def update_best_summary(summary: Any, metrics: Dict[str, float]) -> None:
  """Write ``best_val_loss`` / ``best_test_pass@1_k1`` / ``best_test_gsm8k_pass@1_k1``."""
  if summary is None:
    return

  def _min_key(metric_key: str, summary_key: str) -> None:
    if metric_key not in metrics:
      return
    value = float(metrics[metric_key])
    prev = summary.get(summary_key)
    if prev is None or (not math.isnan(value) and value < float(prev)):
      summary[summary_key] = value

  def _max_key(metric_key: str, summary_key: str) -> None:
    if metric_key not in metrics:
      return
    value = float(metrics[metric_key])
    prev = summary.get(summary_key)
    if prev is None or (not math.isnan(value) and value > float(prev)):
      summary[summary_key] = value

  _min_key("val/loss", "best_val_loss")
  _max_key("test/pass@1_k1", "best_test_pass@1_k1")
  _max_key("test/gsm8k/pass@1_k1", "best_test_gsm8k_pass@1_k1")


def save_protocol_artifacts(
  save_dir: Path,
  step: int,
  tokens: Optional[Any],
  records: Sequence[Dict[str, Any]],
  metrics: Dict[str, float],
) -> Dict[str, str]:
  save_dir.mkdir(parents=True, exist_ok=True)
  json_path = save_dir / f"step_{step}.json"
  payload = {"step": step, "metrics": metrics, "records": list(records)}
  with open(json_path, "w", encoding="utf-8") as fp:
    json.dump(payload, fp, indent=2)
  paths = {"json": str(json_path)}
  if tokens is not None:
    import torch
    pt_path = save_dir / f"step_{step}.pt"
    torch.save(tokens.detach().cpu(), pt_path)
    paths["pt"] = str(pt_path)
  return paths
