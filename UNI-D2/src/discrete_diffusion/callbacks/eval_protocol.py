"""Test-task logging for docs/research/metrics.md (test/* and perf/test_s)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import torch

from ..evaluations.protocol import (
  save_protocol_artifacts,
  update_best_summary,
)
from ..evaluations.task import (
  aggregate_task_metrics,
  encode_question_prefixes,
  infer_task_name,
  load_task_split,
  score_gsm8k_python_record,
  score_gsm8k_record,
  score_tinygsm_record,
)


class EvalProtocol(L.Callback):
  """Log protocol test/* task scalars after each val epoch."""

  def __init__(
      self,
      enabled: bool = True,
      task: Optional[str] = None,
      transfer_task: Optional[str] = None,
      task_num_samples: int = 32,
      wandb_table_rows: int = 16,
      save_dir: str = "./val_samples",
      save_artifacts: bool = True,
  ) -> None:
    super().__init__()
    self.enabled = enabled
    self.task_override = task
    self.transfer_task_override = transfer_task
    self.task_num_samples = task_num_samples
    self.wandb_table_rows = wandb_table_rows
    self.save_dir = Path(save_dir)
    self.save_artifacts = save_artifacts

  def on_validation_epoch_end(self, trainer, pl_module) -> None:
    # Rank 0 runs prefix sampling + exec scoring; other ranks must wait or
    # DDP hits an NCCL ALLREDUCE timeout on the next training step.
    try:
      self._run_rank0_eval(trainer, pl_module)
    finally:
      if int(getattr(trainer, "world_size", 1) or 1) > 1:
        trainer.strategy.barrier()

  def _run_rank0_eval(self, trainer, pl_module) -> None:
    if not self.enabled:
      return
    config = getattr(pl_module, "config", None)
    if config is None:
      return
    if trainer.sanity_checking and not getattr(
        config.eval, "compute_perplexity_on_sanity", False):
      return
    if not getattr(config.eval, "generate_samples", True):
      return
    if not trainer.is_global_zero:
      return

    task = self._resolve_task(config)
    transfer = self._resolve_transfer(config, task)
    create_sampler = getattr(pl_module, "_create_sampler", None)
    sampler = create_sampler() if callable(create_sampler) else None
    supports_prefix = False
    if sampler is not None:
      import inspect
      supports_prefix = "prefix" in inspect.signature(sampler.generate).parameters
    eval_tasks = [t for t in (task, transfer) if t]
    if eval_tasks and not supports_prefix:
      print(
        f"Eval protocol: {type(sampler).__name__} has no prefix sampling; "
        "skipping test/*.")
      eval_tasks = []
    try:
      t0 = time.perf_counter()
      tokens, texts, records, metrics = self._evaluate(
        pl_module, config, eval_tasks)
      if eval_tasks:
        metrics["perf/test_s"] = time.perf_counter() - t0
    except Exception as exc:
      print(f"Eval protocol failed at step {pl_module.global_step}: {exc}")
      return

    for key, value in metrics.items():
      pl_module.log(
        key, value, on_step=False, on_epoch=True,
        sync_dist=False, rank_zero_only=True)

    logger = trainer.logger
    # if texts and hasattr(logger, "log_table"):
    #   logger.log_table(
    #     key="val/samples",
    #     columns=WANDB_TABLE_COLUMNS,
    #     data=wandb_table_rows(records, max_rows=self.wandb_table_rows),
    #   )
    summary = getattr(getattr(logger, "experiment", None), "summary", None)
    callback_metrics = {
      k: float(v.item() if hasattr(v, "item") else v)
      for k, v in trainer.callback_metrics.items()
    }
    callback_metrics.update(metrics)
    update_best_summary(summary, callback_metrics)

    if self.save_artifacts:
      save_protocol_artifacts(
        self.save_dir, int(pl_module.global_step), tokens, records, metrics)

  def _resolve_task(self, config) -> Optional[str]:
    explicit = self.task_override or getattr(config.eval, "task", None)
    try:
      from omegaconf import OmegaConf
      if OmegaConf.is_none(explicit):
        explicit = None
    except Exception:
      pass
    if explicit in {None, "", "null", "none"}:
      return infer_task_name(getattr(config.data, "train", None)) or infer_task_name(
        getattr(config.data, "valid", None))
    name = str(explicit)
    if name.lower() in {"null", "none"}:
      return infer_task_name(getattr(config.data, "train", None)) or infer_task_name(
        getattr(config.data, "valid", None))
    return name

  def _resolve_transfer(self, config, primary: Optional[str]) -> Optional[str]:
    explicit = self.transfer_task_override
    if explicit is None:
      explicit = getattr(config.eval, "transfer_task", None)
    try:
      from omegaconf import OmegaConf
      if OmegaConf.is_none(explicit):
        explicit = None
    except Exception:
      pass
    if explicit in {None, "", "null", "none"}:
      if primary == "tinygsm":
        return "gsm8k"
      return None
    name = str(explicit)
    if name.lower() in {"null", "none"}:
      return "gsm8k" if primary == "tinygsm" else None
    if primary and name == primary:
      return None
    return name

  def _evaluate(self, pl_module, config, eval_tasks: List[str]):
    tokenizer = pl_module.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if not eval_tasks:
      num_batches = int(getattr(config.sampling, "num_sample_batches", 1) or 1)
      batch_size = int(config.loader.eval_batch_size)
      chunks = []
      for _ in range(max(1, num_batches)):
        sample = pl_module.generate_samples(num_samples=batch_size)
        chunks.append(sample.detach().cpu())
      tokens = torch.cat(chunks, dim=0)
      texts = tokenizer.batch_decode(tokens.tolist(), skip_special_tokens=True)
      if pad_id is None:
        lengths = [int(row.numel()) for row in tokens]
      else:
        lengths = [int((row != pad_id).sum().item()) for row in tokens]
      records = [
        {
          "prompt": "",
          "generation": text,
          "gold": "",
          "extracted_answer": None,
          "correct": None,
          "length": int(length),
        }
        for text, length in zip(texts, lengths)
      ]
      return tokens, texts, records, {}

    all_tokens = []
    all_texts: List[str] = []
    all_records: List[Dict[str, Any]] = []
    metrics: Dict[str, float] = {}
    for i, task_name in enumerate(eval_tasks):
      namespace = None if i == 0 else task_name
      tokens, texts, records = self._task_samples(pl_module, config, task_name)
      all_tokens.append(tokens)
      all_texts.extend(texts)
      tagged = []
      for rec in records:
        row = dict(rec)
        row["eval_task"] = task_name
        tagged.append(row)
      all_records.extend(tagged)
      metrics.update(aggregate_task_metrics(records, task_name, namespace))
    cat = torch.cat(all_tokens, dim=0) if all_tokens else None
    return cat, all_texts, all_records, metrics

  def _task_samples(self, pl_module, config, task):
    n = int(self.task_num_samples)
    questions, golds = load_task_split(task, n, config=config)
    tokenizer = pl_module.tokenizer
    batch_size = int(config.loader.eval_batch_size)
    all_tokens = []
    texts: List[str] = []
    records: List[Dict[str, Any]] = []
    for start in range(0, len(questions), batch_size):
      qs = questions[start:start + batch_size]
      gs = golds[start:start + batch_size]
      prefixes = encode_question_prefixes(tokenizer, qs)
      samples = pl_module.generate_samples(
        num_samples=len(prefixes), prefix=prefixes)
      samples = samples.detach().cpu()
      all_tokens.append(samples)
      decoded = tokenizer.batch_decode(samples.tolist(), skip_special_tokens=True)
      pad_id = tokenizer.pad_token_id
      for i, gen in enumerate(decoded):
        gold = gs[i]
        match = score_gsm8k_record(gen, gold)
        if task == "tinygsm":
          py = score_tinygsm_record(gen, gold)
        else:
          py = score_gsm8k_python_record(gen, gold)
        prompt_ids = prefixes[i]
        gen_ids = samples[i]
        length = int((gen_ids != pad_id).sum().item()) if pad_id is not None else int(gen_ids.numel())
        primary = match if task == "gsm8k" else py
        records.append({
          "prompt": qs[i],
          "generation": gen,
          "gold": primary.get("gold", gold),
          "gold_answer": gold,
          "extracted_answer": primary.get("extracted_answer"),
          "correct": primary.get("correct"),
          "extracted": primary.get("extracted"),
          "correct_match": match.get("correct"),
          "extracted_match": match.get("extracted"),
          "correct_pass1": py.get("correct"),
          "exec_error": py.get("exec_error"),
          "length": length,
          "prompt_len": int(prompt_ids.numel()),
        })
        texts.append(gen)
    tokens = torch.cat(all_tokens, dim=0)
    return tokens, texts, records


class TaskMetricCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
  """ModelCheckpoint that no-ops until the configured task metric is logged."""

  def on_validation_end(self, trainer, pl_module) -> None:
    if self.monitor not in trainer.callback_metrics:
      return
    super().on_validation_end(trainer, pl_module)
