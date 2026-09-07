"""Pretokenize slim APPS/TACO (question + answer) into PUMA memmaps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm.auto import tqdm

_UNI_D2_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_UNI_D2_SRC) not in sys.path:
    sys.path.insert(0, str(_UNI_D2_SRC))


def pretokenize_code_qa(
    out_dir: str,
    source_dir: str,
    tokenizer_name: str = "Qwen/Qwen2-0.5B",
    max_len: int = 512,
    sep: str = "\n",
    split: str = "train",
):
    if max_len % 8 != 0:
        raise ValueError("max_len must be divisible by 8")
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    sep_ids = tok(sep, add_special_tokens=False).input_ids
    ds = load_from_disk(source_dir)[split]
    n = len(ds)
    labels_path = os.path.join(out_dir, "labels.bin")
    mask_path = os.path.join(out_dir, "prompt_mask.bin")
    labels = np.memmap(labels_path, mode="w+", dtype=np.uint32, shape=(n, max_len))
    mask = np.memmap(mask_path, mode="w+", dtype=np.uint8, shape=(n, max_len // 8))
    labels[:] = eos_id
    batch = 512
    for start in tqdm(range(0, n, batch), desc=f"Pretokenizing -> {out_dir}"):
        end = min(start + batch, n)
        rows = ds[start:end]
        prompts = [(q or "").strip() for q in rows["question"]]
        answers = [(a or "").strip() for a in rows["answer"]]
        p_batch = tok(prompts, add_special_tokens=False).input_ids
        a_batch = tok(answers, add_special_tokens=False).input_ids
        b = end - start
        ids_arr = np.full((b, max_len), eos_id, dtype=np.uint32)
        pm = np.zeros((b, max_len), dtype=np.uint8)
        for i, (p_ids, a_ids) in enumerate(zip(p_batch, a_batch)):
            raw = p_ids + sep_ids + a_ids
            prompt_len = len(p_ids) + len(sep_ids)
            if len(raw) >= max_len:
                ids_arr[i, : max_len - 1] = raw[: max_len - 1]
                boundary = min(prompt_len, max_len - 1)
            else:
                ids_arr[i, : len(raw)] = raw
                boundary = min(prompt_len, max_len)
            pm[i, :boundary] = 1
        labels[start:end] = ids_arr
        mask[start:end] = np.packbits(pm, axis=-1, bitorder="little")
    labels.flush()
    mask.flush()
    meta = {
        "dataset": os.path.basename(source_dir.rstrip("/")),
        "split": split,
        "tokenizer": tokenizer_name,
        "max_len": max_len,
        "sep": sep,
        "eos_id": int(eos_id),
        "num_examples": int(n),
        "labels_dtype": "uint32",
        "prompt_mask_packed": True,
        "prompt_mask_bitorder": "little",
        "source_dir": source_dir,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {n} examples to {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--source_dir", required=True,
                        help="Slim APPS/TACO DatasetDict from load_code_qa_dataset")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2-0.5B")
    args = parser.parse_args()
    pretokenize_code_qa(
        args.out_dir, args.source_dir,
        tokenizer_name=args.tokenizer, max_len=args.max_len)
