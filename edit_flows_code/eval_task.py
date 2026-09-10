"""GSM8K / TinyGSM / APPS / TACO scoring for Edit Flows."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from discrete_diffusion.evaluations.code_exec import score_apps_record
except ImportError:
    score_apps_record = None

GSM8K_ANSWER_RE = re.compile(r"####\s*([-\d.,]+)")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def normalize_number(text: Optional[str]) -> Optional[str]:
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
    matches = GSM8K_ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return normalize_number(matches[-1])


def _code_to_run(code: str) -> str:
    text = code or ""
    idx = text.find("def simple_math_problem")
    if idx < 0:
        idx = text.find("def ")
    return text[idx:] if idx >= 0 else text


def execute_tinygsm(code: str, timeout: float = 5.0) -> Optional[str]:
    harness = _code_to_run(code) + "\n\nprint(simple_math_problem())\n"
    try:
        proc = subprocess.run(
            ["python", "-c", harness], capture_output=True, text=True, timeout=timeout,
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


def score_tinygsm_record(generation: str, gold: str, timeout: float = 5.0) -> Dict[str, Any]:
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


def score_gsm8k_python_record(generation: str, gold: str, timeout: float = 5.0) -> Dict[str, Any]:
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


DEFAULT_PASS_AT_1_K = (1, 2, 4, 8)


def aggregate_task_metrics(
    records: Sequence[Dict[str, Any]],
    task: str = "gsm8k",
    key_namespace: Optional[str] = None,
    k: int = 1,
) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    if any("correct_pass1" in rec for rec in records):
        pass1_ok = sum(1 for rec in records if rec.get("correct_pass1"))
    else:
        pass1_ok = sum(1 for rec in records if rec.get("correct"))
    metrics = {f"test/pass@1_k{int(k)}": pass1_ok / n}
    return namespace_test_metrics(metrics, key_namespace)
