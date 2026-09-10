"""Local checks for eval protocol parsers (tests/ is gitignored)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "UNI-D2" / "src" / "discrete_diffusion" / "evaluations"
sys.path.insert(0, str(EVAL))


def _load(mod_name: str, filename: str):
  spec = importlib.util.spec_from_file_location(mod_name, EVAL / filename)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


task = _load("eval_task", "task.py")
protocol = _load("eval_protocol_mod", "protocol.py")


def test_gsm8k_extract_last_marker():
  text = "scratch work #### 12\ntherefore #### 84"
  assert task.extract_gsm8k_answer(text) == "84"


def test_gsm8k_normalize_commas_and_int():
  assert task.extract_gsm8k_answer("#### 1,024") == "1024"
  assert task.normalize_number("3.0") == "3"


def test_gsm8k_score_match_and_extract_frac():
  gold = "The answer is #### 42"
  ok = task.score_gsm8k_record("reasoning\n#### 42", gold)
  bad = task.score_gsm8k_record("no marker here", gold)
  miss = task.score_gsm8k_record("#### 7", gold)
  assert ok["correct"] and ok["extracted"]
  assert not bad["correct"] and not bad["extracted"]
  assert miss["extracted"] and not miss["correct"]
  metrics = task.aggregate_task_metrics([ok, bad, miss], "gsm8k", k=1)
  assert metrics["test/pass@1_k1"] == 1 / 3
  assert "test/pass@1_match" not in metrics
  assert "test/pass@1" not in metrics


def test_tinygsm_exec_and_namespace():
  gold = "def simple_math_problem():\n    return 6\n"
  ok = task.score_tinygsm_record(gold, gold)
  bad = task.score_tinygsm_record("print('nope')", gold)
  assert ok["correct"] and not ok["exec_error"]
  assert not bad["correct"] and bad["exec_error"]
  records = [
    {**ok, "correct_match": False, "extracted_match": False, "correct_pass1": ok["correct"]},
    {**bad, "correct_match": False, "extracted_match": False, "correct_pass1": bad["correct"]},
  ]
  in_domain = task.aggregate_task_metrics(records, "tinygsm", k=1)
  assert in_domain["test/pass@1_k1"] == 0.5
  transfer = task.aggregate_task_metrics(records, "gsm8k-python", key_namespace="gsm8k", k=2)
  assert transfer["test/gsm8k/pass@1_k2"] == 0.5
  assert "test/pass@1" not in transfer
  assert "test/pass@1_k1" not in transfer


def test_tinygsm_cache_path():
  path = task.tinygsm_raw_cache_path("/data/scratch/tinygsm", 100000, 42)
  assert path.endswith("tinygsm_raw_n100000_seed42.dat")


def test_best_summary():
  summary = {}
  protocol.update_best_summary(summary, {"val/loss": 2.5, "test/pass@1_k1": 0.1})
  protocol.update_best_summary(summary, {"val/loss": 2.0, "test/pass@1_k1": 0.4})
  protocol.update_best_summary(summary, {"val/loss": 2.2, "test/pass@1_k1": 0.3})
  protocol.update_best_summary(summary, {"test/gsm8k/pass@1_k1": 0.2})
  assert summary["best_val_loss"] == 2.0
  assert summary["best_test_pass@1_k1"] == 0.4
  assert summary["best_test_gsm8k_pass@1_k1"] == 0.2
  assert "best_val_nll" not in summary
  assert "best_gen_ppl" not in summary
  assert "best_test_pass@1_match" not in summary


if __name__ == "__main__":
  test_gsm8k_extract_last_marker()
  test_gsm8k_normalize_commas_and_int()
  test_gsm8k_score_match_and_extract_frac()
  test_tinygsm_exec_and_namespace()
  test_tinygsm_cache_path()
  test_best_summary()
  print("ok")
