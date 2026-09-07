#!/usr/bin/env python3
"""Flatten a comparison-repo UNI-D2 method YAML into Hydra CLI overrides.

The YAML ``hydra:`` block is config-group selection (data=, algo=, …), not
Hydra runtime config. ``hydra_run_dir`` becomes ``hydra.run.dir=``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _parse_scalar(text: str):
    text = text.strip()
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text in ("null", "None", "~"):
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_simple(path: Path) -> dict:
    """Indent YAML subset used by configs/mdlm and configs/flexmdm."""
    root: dict = {}
    stack = [(-1, root)]
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if rest == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def _load(path: Path) -> dict:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except ImportError:
        pass
    try:
        import yaml
    except ImportError:
        return _load_simple(path)
    with path.open() as handle:
        return yaml.safe_load(handle)


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt(v) for v in value) + "]"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        if any(ch in value for ch in "/ @,"):
            return f'"{value}"'
        return value
    return str(value)


def _walk(prefix: str, node, out: list) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _walk(next_prefix, value, out)
        return
    out.append(f"{prefix}={_fmt(node)}")


def overrides_from_yaml(path: Path) -> list[str]:
    cfg = _load(path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"expected a mapping in {path}")
    out = []
    for key, value in (cfg.pop("hydra", None) or {}).items():
        out.append(f"{key}={value}")
    run_dir = cfg.pop("hydra_run_dir", None)
    if run_dir is not None:
        out.append(f"hydra.run.dir={run_dir}")
    _walk("", cfg, out)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: hydra_from_yaml.py <yaml>")
    path = Path(sys.argv[1])
    if not path.is_file():
        sys.exit(f"missing config: {path}")
    for line in overrides_from_yaml(path):
        print(line)


if __name__ == "__main__":
    main()
