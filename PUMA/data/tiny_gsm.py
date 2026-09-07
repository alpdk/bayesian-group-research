import os, json
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, random_split


# per-worker state for parallel pretokenization (initialized in _init_worker)
_W = {}

def _init_worker(tokenizer_name, max_len, sep, labels_path, mask_path, N):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    _W["tok"] = tok
    _W["sep_ids"] = tok(sep, add_special_tokens=False).input_ids
    _W["eos"] = tok.eos_token_id
    _W["max_len"] = max_len
    # Arrow cache is memory-mapped, so re-opening per worker is cheap
    _W["ds"] = load_dataset("TinyGSM/TinyGSM", split="train")
    _W["labels"] = np.memmap(labels_path, mode="r+", dtype=np.uint32, shape=(N, max_len))
    _W["mask"] = np.memmap(mask_path, mode="r+", dtype=np.uint8, shape=(N, max_len // 8))


def _process_range(rng):
    """Tokenize dataset rows [start, end) and write them into the shared memmaps.

    Workers write disjoint row ranges, so no locking is needed.
    """
    start, end = rng
    tok, ds = _W["tok"], _W["ds"]
    sep_ids, eos, max_len = _W["sep_ids"], _W["eos"], _W["max_len"]

    for s in range(start, end, 2048):
        e = min(s + 2048, end)
        rows = ds[s:e]
        prompts = [(q or "").strip() for q in rows["question"]]
        answers = [(c or "").strip() for c in rows["code"]]

        p_batch = tok(prompts, add_special_tokens=False).input_ids
        a_batch = tok(answers, add_special_tokens=False).input_ids

        B = e - s
        ids_arr = np.full((B, max_len), eos, dtype=np.uint32)  # EOS doubles as padding
        pm = np.zeros((B, max_len), dtype=np.uint8)

        for i, (p_ids, a_ids) in enumerate(zip(p_batch, a_batch)):
            raw = p_ids + sep_ids + a_ids
            prompt_len = len(p_ids) + len(sep_ids)
            if len(raw) >= max_len:
                # truncate; final position stays EOS from np.full
                ids_arr[i, : max_len - 1] = raw[: max_len - 1]
                boundary = min(prompt_len, max_len - 1)
            else:
                ids_arr[i, : len(raw)] = raw
                boundary = min(prompt_len, max_len)
            pm[i, :boundary] = 1

        _W["labels"][s:e] = ids_arr
        _W["mask"][s:e] = np.packbits(pm, axis=-1, bitorder="little")

    _W["labels"].flush()
    _W["mask"].flush()
    return end - start


def pretokenize_tinygsm(
    out_dir: str,
    tokenizer_name: str = "Qwen/Qwen2-0.5B",
    max_len: int = 512,
    sep: str = "\n",
    num_proc: int | None = None,
    limit: int | None = None,
):
    """
    Tokenize TinyGSM into fixed-length sequences and save as memmaps
    (parallel across num_proc processes; downloads the dataset once via HF cache).

    Sequence format (MDM-style, one example = one seq):
      ids_raw = prompt_ids + sep_ids + answer_ids
      if len(ids_raw) >= max_len:
         ids = ids_raw[:max_len-1] + [EOS]
      else:
         ids = ids_raw + [EOS] * (max_len - len(ids_raw))   # EOS used as padding too

    prompt_mask convention:
      - True for prompt tokens
      - False for answer tokens + pad tokens
    """
    import multiprocessing as mp

    os.makedirs(out_dir, exist_ok=True)
    if max_len % 8 != 0:
        raise ValueError("max_len must be divisible by 8 to pack prompt_mask with packbits cleanly.")
    if num_proc is None:
        num_proc = max(1, os.cpu_count() or 1)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    print("Loading TinyGSM (non-streaming; downloads to HF cache on first use)...")
    ds = load_dataset("TinyGSM/TinyGSM", split="train")
    N = len(ds)
    if limit is not None:
        N = min(N, int(limit))

    labels_path = os.path.join(out_dir, "labels.bin")
    mask_path   = os.path.join(out_dir, "prompt_mask.bin")
    meta_path   = os.path.join(out_dir, "meta.json")

    # Pre-allocate memmaps; workers open them r+ and fill disjoint row ranges
    labels_mm = np.memmap(labels_path, mode="w+", dtype=np.uint32, shape=(N, max_len))
    mask_mm   = np.memmap(mask_path,   mode="w+", dtype=np.uint8,  shape=(N, max_len // 8))
    labels_mm.flush(); mask_mm.flush()
    del labels_mm, mask_mm

    chunk = 131072
    ranges = [(s, min(s + chunk, N)) for s in range(0, N, chunk)]
    num_proc = min(num_proc, len(ranges))

    # spawn: fork is unsafe with rust tokenizers / arrow threads
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        num_proc,
        initializer=_init_worker,
        initargs=(tokenizer_name, max_len, sep, labels_path, mask_path, N),
    ) as pool:
        with tqdm(total=N, desc=f"Pretokenizing TinyGSM -> {out_dir} ({num_proc} procs)") as pbar:
            for done in pool.imap_unordered(_process_range, ranges):
                pbar.update(done)

    meta = {
        "dataset": "TinyGSM/TinyGSM",
        "split": "train",
        "tokenizer": tokenizer_name,
        "max_len": max_len,
        "sep": sep,
        "eos_id": int(eos_id),
        "num_examples": int(N),
        "labels_dtype": "uint32",
        "prompt_mask_packed": True,
        "prompt_mask_bitorder": "little",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote {N:,} examples")
    print(f"- {labels_path} (uint32) shape=({N},{max_len})")
    print(f"- {mask_path}   (packed uint8) shape=({N},{max_len//8})")
    print(f"- {meta_path}")



class TinyGSMDataset(Dataset):
    """
    Loads pretokenized TinyGSM from:
      - labels.bin       uint32 [N, max_len]
      - prompt_mask.bin  uint8  [N, max_len/8] (packed bits)
      - meta.json
    Returns:
      {"labels": LongTensor[max_len], "prompt_mask": BoolTensor[max_len]}
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.max_len = int(self.meta["max_len"])
        self.N = int(self.meta["num_examples"])
        self.bitorder = self.meta.get("prompt_mask_bitorder", "little")

        labels_path = os.path.join(data_dir, "labels.bin")
        mask_path   = os.path.join(data_dir, "prompt_mask.bin")

        self.labels_mm = np.memmap(labels_path, mode="r", dtype=np.uint32, shape=(self.N, self.max_len))
        self.mask_mm   = np.memmap(mask_path,   mode="r", dtype=np.uint8,  shape=(self.N, self.max_len // 8))

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        labels = torch.from_numpy(self.labels_mm[idx].astype(np.int64))  # to torch long
        packed = self.mask_mm[idx]
        mask = np.unpackbits(packed, bitorder=self.bitorder)[: self.max_len].astype(np.bool_)
        prompt_mask = torch.from_numpy(mask)

        return {"labels": labels, "prompt_mask": prompt_mask}

def split_tinygsm(data_dir: str, val_ratio: float = 0.05, seed: int = 2025):
    dataset = TinyGSMDataset(data_dir)
    n = len(dataset)

    n_val = int(n * val_ratio)
    n_train = n - n_val

    g = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(dataset, [n_train, n_val], generator=g)
    return train_data, val_data



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/tiny_gsm")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=None)
    args = parser.parse_args()
    pretokenize_tinygsm(out_dir=args.out_dir, max_len=args.max_len, limit=args.limit, num_proc=args.num_proc)