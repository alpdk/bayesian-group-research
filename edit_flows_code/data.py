"""Dataset loading for edit-flow code generation.

Supported datasets (prompt -> target):
- gsm8k:   openai/gsm8k question -> chain-of-thought answer ending in "#### <num>"
- tinygsm: TinyGSM/TinyGSM question -> Python solution (def simple_math_problem())
- apps:    codeparrot/apps question (+ starter) -> first Python solution
- taco:    BAAI/TACO question (+ starter) -> first Python solution
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from tokenizer import ByteTokenizer

DATASET_CHOICES = ("gsm8k", "tinygsm", "apps", "taco")

_DOCSTRING_RE = re.compile(r'\n\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', flags=re.MULTILINE)


def strip_docstring(code: str) -> str:
    """Removes the docstring from TinyGSM solutions (it just repeats the question)."""
    return _DOCSTRING_RE.sub("", code, count=1)


def _uni_d2_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def _load_code_qa(dataset: str, split: str):
    src = _uni_d2_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from discrete_diffusion.data.loaders import load_code_qa_dataset

    cache_root = os.environ.get(
        "DISCRETE_DIFFUSION_SCRATCH_DIR",
        str(Path.home() / ".cache" / "discrete_diffusion"),
    )
    cache_dir = os.path.join(cache_root, dataset)
    ds = load_code_qa_dataset(dataset, cache_dir)
    key = "test" if split in {"test", "validation"} else "train"
    return ds[key]


def load_pairs(
    dataset: str,
    split: str = "train",
    max_examples: int | None = None,
    keep_docstring: bool = False,
) -> List[Tuple[str, str]]:
    from datasets import load_dataset

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        return [(ex["question"].strip(), ex["answer"].strip()) for ex in ds]

    if dataset == "tinygsm":
        assert split == "train", "TinyGSM only has a train split"
        assert max_examples is not None, "TinyGSM is huge (11.8M rows); pass max_examples"
        ds = load_dataset("TinyGSM/TinyGSM", split="train", streaming=True)
        pairs = []
        for ex in ds:
            code = ex["code"] if keep_docstring else strip_docstring(ex["code"])
            pairs.append((ex["question"].strip(), code.strip()))
            if len(pairs) >= max_examples:
                break
        return pairs

    if dataset in {"apps", "taco"}:
        ds = _load_code_qa(dataset, split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        pairs = []
        for ex in ds:
            answer = (ex.get("answer") or "").strip()
            if split != "train" and not answer:
                continue
            pairs.append(((ex.get("question") or "").strip(), answer))
        return pairs

    raise ValueError(f"Unknown dataset {dataset!r}, choose from {DATASET_CHOICES}")


def load_eval_pairs(
    dataset: str,
    max_examples: int | None = None,
) -> List[Tuple[str, str]]:
    """Prompt + gold used for task scoring (input_output JSON for APPS/TACO)."""
    if dataset in {"apps", "taco"}:
        ds = _load_code_qa(dataset, "test")
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        return [
            ((ex.get("question") or "").strip(), ex.get("input_output") or "")
            for ex in ds
        ]
    if dataset == "tinygsm":
        return load_pairs("tinygsm", split="train", max_examples=max_examples or 256)
    return load_pairs("gsm8k", split="test", max_examples=max_examples)


class CodeGenDataset:
    """Tokenized (prompt, target) pairs with length filtering and batch sampling."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: ByteTokenizer,
        max_prompt_len: int = 512,
        max_target_len: int = 512,
        filter_target: bool = True,
    ):
        self.tokenizer = tokenizer
        self.prompts: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.texts: List[Tuple[str, str]] = []
        n_skipped = 0
        for prompt_text, target_text in pairs:
            p = tokenizer.encode(prompt_text)
            x = tokenizer.encode(target_text)
            if len(p) > max_prompt_len or len(x) == 0 or (
                    filter_target and len(x) > max_target_len):
                n_skipped += 1
                continue
            self.prompts.append(p)
            self.targets.append(x)
            self.texts.append((prompt_text, target_text))
        print(f"Dataset: kept {len(self.prompts)} examples, "
              f"skipped {n_skipped} (prompt > {max_prompt_len} or target > {max_target_len} bytes)")
        assert len(self.prompts) > 0, "All examples were filtered out; increase length limits"

    def __len__(self):
        return len(self.prompts)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.prompts), size=batch_size)
        return [self.prompts[i] for i in idx], [self.targets[i] for i in idx]


def pad_stack(seqs: List[torch.Tensor], pad_token: int) -> torch.Tensor:
    """Right-pads a list of 1D tensors to a (batch, max_len) tensor."""
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_token, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out
