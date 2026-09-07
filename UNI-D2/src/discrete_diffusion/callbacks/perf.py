"""Ongoing train/val step timing under ``perf/`` (docs/research/metrics.md)."""

from __future__ import annotations

import time
from typing import Any

import lightning as L


def _num_tokens(batch: Any) -> int:
  if not isinstance(batch, dict):
    return 0
  mask = batch.get("attention_mask")
  if mask is not None:
    return int(mask.sum().item())
  ids = batch.get("input_ids")
  if ids is not None:
    return int(ids.numel())
  return 0


class PerfMonitor(L.Callback):
  """Log train/val wall-time throughput under ``perf/``."""

  def __init__(self, enabled: bool = True) -> None:
    super().__init__()
    self.enabled = enabled
    self._train_t0: float | None = None
    self._val_t0: float | None = None

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
    if self.enabled:
      self._train_t0 = time.perf_counter()

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
    self._log(pl_module, batch, self._train_t0, split="train", on_step=True)
    self._train_t0 = None

  def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
    if self.enabled:
      self._val_t0 = time.perf_counter()

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
    self._log(pl_module, batch, self._val_t0, split="val", on_step=False)
    self._val_t0 = None

  def _log(self, pl_module, batch, t0, split: str, on_step: bool) -> None:
    if not self.enabled or t0 is None:
      return
    elapsed = time.perf_counter() - t0
    if elapsed <= 0:
      return
    tokens = _num_tokens(batch)
    world = getattr(pl_module.trainer, "world_size", 1) or 1
    kwargs = dict(on_step=on_step, on_epoch=True, sync_dist=False, rank_zero_only=True)
    pl_module.log(f"perf/{split}_step_s", elapsed, **kwargs)
    if tokens > 0:
      pl_module.log(
        f"perf/{split}_tokens_per_s", tokens * world / elapsed, **kwargs)
