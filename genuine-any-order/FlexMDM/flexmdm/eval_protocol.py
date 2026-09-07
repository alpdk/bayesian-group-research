"""W&B keys from UNI-D2 docs/eval_metrics.md."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def nll_family(nll: float) -> Dict[str, float]:
    nll = float(nll)
    return {
        "val/nll": nll,
    }


def update_best_summary(summary: Any, metrics: Dict[str, float]) -> None:
    if summary is None:
        return
    nll = metrics.get("val/nll")
    if nll is not None:
        prev = summary.get("best_val_nll")
        if prev is None or float(nll) < float(prev):
            summary["best_val_nll"] = float(nll)
    exec_acc = metrics.get("test/pass@1")
    if exec_acc is not None:
        prev = summary.get("best_test_pass@1")
        if prev is None or float(exec_acc) > float(prev):
            summary["best_test_pass@1"] = float(exec_acc)
    exact = metrics.get("test/pass@1_match")
    if exact is not None:
        prev = summary.get("best_test_pass@1_match")
        if prev is None or float(exact) > float(prev):
            summary["best_test_pass@1_match"] = float(exact)


def update_early_stop(
    nll: Optional[float],
    best: Optional[float],
    stale: int,
    *,
    patience: int = 5,
    min_delta: float = 0.0,
) -> tuple[bool, Optional[float], int]:
    """Lightning-style EarlyStopping on ``val/nll`` (min, check_finite)."""
    if nll is None:
        return False, best, stale
    if not math.isfinite(float(nll)):
        return True, best, stale
    nll_f = float(nll)
    if best is None or nll_f < float(best) - min_delta:
        return False, nll_f, 0
    stale += 1
    return stale >= max(int(patience), 1), best, stale


def extract_task_exact_match(val_acc: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not val_acc:
        return {}
    for key, value in val_acc.items():
        if isinstance(value, dict) and "pass_at_1" in value:
            acc = float(value["pass_at_1"])
            return {"test/pass@1_match": acc, "test/pass@1": acc}
        if isinstance(value, (int, float)) and "pass_at_1" in str(key):
            acc = float(value)
            return {"test/pass@1_match": acc, "test/pass@1": acc}
    for value in val_acc.values():
        if isinstance(value, (int, float)):
            acc = float(value)
            return {"test/pass@1_match": acc, "test/pass@1": acc}
    return {}
