"""Score saved sample JSON against the eval protocol (no GPU).

Accepts Edit Flows ``samples.json`` (list of records) or protocol artifacts
(``{"records": [...]}``).

Example:
  .venv/bin/python -m discrete_diffusion.evaluations.score_samples \\
      --samples path/to/samples.json --task gsm8k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .task import (
  aggregate_task_metrics,
  score_gsm8k_record,
  score_tinygsm_record,
)


def _load_records(path: Path) -> List[Dict[str, Any]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(payload, list):
    return payload
  if isinstance(payload, dict) and "records" in payload:
    return list(payload["records"])
  raise ValueError(f"Unsupported JSON schema in {path}")


def _generation(rec: Dict[str, Any]) -> str:
  return rec.get("generation") or rec.get("text") or ""


def _gold(rec: Dict[str, Any]) -> str:
  return rec.get("gold_answer") or rec.get("gold") or rec.get("answer") or ""


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--samples", type=str, required=True)
  parser.add_argument("--task", type=str, choices=("gsm8k", "tinygsm"), default="gsm8k")
  parser.add_argument("--timeout", type=float, default=5.0)
  args = parser.parse_args()

  path = Path(args.samples)
  raw = _load_records(path)
  scored = []
  for rec in raw:
    gen = _generation(rec)
    gold = _gold(rec)
    if args.task == "tinygsm":
      row = score_tinygsm_record(gen, gold, timeout=args.timeout)
    else:
      row = score_gsm8k_record(gen, gold)
    row.update({
      "prompt": rec.get("question") or rec.get("prompt") or "",
      "generation": gen,
    })
    scored.append(row)

  metrics = aggregate_task_metrics(scored, args.task)
  print(json.dumps({"file": str(path), "n": len(scored), **metrics}, indent=2))


if __name__ == "__main__":
  main()
