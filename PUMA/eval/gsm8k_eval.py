import torch
import os
import io
import contextlib
import signal
import numpy as np
import math
import torch.distributed as dist
import re
import json
import warnings
from transformers import AutoTokenizer
from sampling import mdm_sampling, mdm_sampling_block, arm_sampling
from tqdm import tqdm
from datasets import load_dataset


# -----------------------------
# Tokenizer cache (Qwen2)
# -----------------------------
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def get_sep_ids():
    tok = get_tokenizer()
    return tok(SEP, add_special_tokens=False).input_ids

TOKENIZER_NAME = "Qwen/Qwen2-0.5B"
MAX_LEN = 512
SEP = "\n"
MASK_ID = 151644

# SEP__ID: 198
# PAD__ID: 151643
# EOS__ID: 151643
# model_vocab_size: 151645

# -----------------------------
# GSM8K answer parsing
# -----------------------------

_ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
def extract_gsm8k_final_answer(ans_text: str) -> str:
    """
    GSM8K 'answer' field includes reasoning and ends with: '#### 72'
    Returns the numeric string ('72', '-3', '1,234', '10.5', etc.)
    """
    m = _ANS_RE.search(ans_text)
    if not m:
        # fallback: try last number in string
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", ans_text)
        return nums[-1].replace(",", "") if nums else ""
    return m.group(1).replace(",", "")

def test_gsm8k_tokenization(mask_id: int, max_len: int = MAX_LEN):
    """
    Creates/loads:
      data/gsm8k_test/test_mdm.json           (max_len == 512, bundled cache)
      data/gsm8k_test/test_mdm_{max_len}.json (other lengths)
    Format:
        inpus_ids: [question_ids] [SEP] [MASK] ...
        answer: numeric string
    Returns:
        X: np.ndarray[num_test, max_len]
        answers: list[num_test]
    """
    fname = "test_mdm.json" if max_len == 512 else f"test_mdm_{max_len}.json"
    out_path = os.path.join("data", "gsm8k_test", fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def load_cached():
        with open(out_path, "r") as f:
            records = json.load(f)
        X = np.array([r["input_ids"] for r in records], dtype=np.int64)
        answers = [r["answer"] for r in records]
        return X, answers

    if os.path.exists(out_path):
        return load_cached()

     # ---- DDP-safe build: only rank0 writes ----
    ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp else 0

    if ddp and rank != 0:
        dist.barrier()
        return load_cached()

    tokenizer = get_tokenizer()
    sep_ids = get_sep_ids()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    records = []

    for ex in ds:
        q = (ex.get("question") or "").strip()
        a = ex.get("answer") or ""

        gold = extract_gsm8k_final_answer(a)

        q_ids = tokenizer(q, add_special_tokens=False).input_ids
        prompt_ids = q_ids + sep_ids

        # Ensure we always leave at least 1 token for mask region
        if len(prompt_ids) >= max_len:
            prompt_ids = prompt_ids[: max_len - 1]

        ids = prompt_ids + [mask_id] * (max_len - len(prompt_ids))

        records.append({
            "input_ids": ids,
            "answer": gold
        })

    # Atomic-ish write to avoid partial files
    tmp_path = out_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(records, f)
    os.replace(tmp_path, out_path)

    # release the non-zero ranks waiting in the barrier above
    if ddp:
        dist.barrier()

    return load_cached()


def _gold_number_from_code(code: str) -> str:
    try:
        with _time_limit(1.0):
            ns = _safe_exec_no_timer(_extract_code(code))
            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return ""
            return str(_to_number(fn()))
    except (_Timeout, Exception):
        return ""


def evaluate_ddp_tinygsm_packed(model, cfg, device, rank: int, world_size: int, sampling):
    """Generate on packed TinyGSM rows in ``data_dir`` (train=eval overfit)."""
    from data.tiny_gsm import TinyGSMDataset

    mask_id = cfg.data.mask_id
    eos_id = int(getattr(cfg.training, "eos_id", 151643))
    ds = TinyGSMDataset(cfg.data.data_dir)
    N_val = len(ds)
    per_rank = math.ceil(N_val / max(world_size, 1))
    start = rank * per_rank
    end = min(start + per_rank, N_val)
    tokenizer = get_tokenizer()
    local_correct, local_total = 0, 0
    local_rows = []

    with torch.no_grad():
        for i in range(start, end):
            item = ds[i]
            labels = item["labels"].to(device)
            prompt_mask = item["prompt_mask"].to(device)
            xt = torch.where(
                prompt_mask, labels, torch.full_like(labels, mask_id))
            samples_tensor = mdm_sampling(
                model, xt.unsqueeze(0), mask_id, sampling, device,
                arm_init=cfg.model.arm_init != "none")
            samples_tensor = samples_tensor.masked_fill(
                samples_tensor == mask_id, tokenizer.pad_token_id)
            gen = tokenizer.decode(
                samples_tensor[0].cpu().tolist(), skip_special_tokens=True)
            gold_ids = labels[(~prompt_mask) & (labels != eos_id)].cpu().tolist()
            gold = tokenizer.decode(gold_ids, skip_special_tokens=True)
            gold_n = _gold_number_from_code(gold)
            ok = bool(gold_n) and evaluate_samples(gen, gold_n)
            local_correct += int(ok)
            local_total += 1
            local_rows.append((ok, gold_n, gen))

    tensor = torch.tensor(
        [local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()
    if rank == 0:
        print(
            f"TinyGSM packed eval: {global_correct}/{global_total} "
            f"pass@1={global_correct / max(global_total, 1):.3f}")
        for i, (ok, gold_n, gen) in enumerate(local_rows[:8]):
            print(
                f"  [{i}] correct={ok} gold={gold_n!r} "
                f"gen={gen.replace(chr(10), ' ')[:240]!r}")
    return global_correct / max(global_total, 1)


def evaluate_ddp_gsm8k(model, cfg, device, rank: int, world_size: int, sampling):
    mask_id = cfg.data.mask_id
    diffusion_layout = getattr(cfg.training, "diffusion_layout", None)
    if diffusion_layout not in {"full", "block"}:
        raise ValueError(
            "training.diffusion_layout must be either 'full' or 'block', "
            f"got {diffusion_layout!r}"
        )

    # pre-tokenize the gsm8k test set (cached) at the model's sequence length
    max_len = int(cfg.model.max_position)
    X, answers = test_gsm8k_tokenization(mask_id, max_len=max_len)
    N_val = len(X)

    # distribute test cases
    per_rank = math.ceil(N_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, N_val)

    batch_size = 32
    num_batches = math.ceil((end - start) / batch_size)
    local_correct, local_total = 0, 0

    tokenizer = get_tokenizer()

    with torch.no_grad():
        for j in tqdm(range(num_batches), desc = "Evaluating"):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X = torch.from_numpy(X[s:e]).long().to(device)
            batch_answers = answers[s:e]
            if diffusion_layout == "block":
                block_size = cfg.training.block_size
                samples_tensor = mdm_sampling_block(model, batch_X, block_size, mask_id, sampling, device)
            elif cfg.training.strategy == "arm":
                samples_tensor = arm_sampling(model, batch_X, mask_id, sampling, device)
            else:
                samples_tensor = mdm_sampling(model, batch_X, mask_id, sampling, device, arm_init=cfg.model.arm_init!="none")

            # tokenizer by default doesn't have mask_id
            samples_tensor = samples_tensor.masked_fill(samples_tensor == mask_id, tokenizer.pad_token_id)

            # sample preproceessing, and extract the answer part
            sample_ids = samples_tensor.cpu().numpy()
            samples = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
            
            for sample, answer in zip(samples, batch_answers):
                if evaluate_samples(sample, answer):
                    local_correct += 1
                local_total += 1
    
    # accumulate succcess rates
    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()
    return global_correct / global_total


def evaluate_samples(sample: str, answer: str, timeout_s: float = 1.0) -> bool:
    """
    sample: model output (string)
    answer: GSM8K answer (string)

    Key fix:
      - timeout now covers BOTH exec(code) and fn() execution
      - prevents a single pathological sample from hanging a rank forever
    """
    code = _extract_code(sample)

    try:
        with _time_limit(timeout_s):
            ns = _safe_exec_no_timer(code)

            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return False

            out = fn()  # now time-bounded

    except (_Timeout, Exception):
        return False

    pred = _to_number(out)
    gold = _to_number(answer)
    return _numbers_equal(pred, gold)

# -----------------------------
# Code execution functions
# -----------------------------
class _Timeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _Timeout()


@contextlib.contextmanager
def _time_limit(timeout_s: float):
    """
    Hard wall-clock time limit using SIGALRM/ITIMER_REAL (POSIX).
    Note: works only in the main thread of the process.
    """
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    old_handler = None
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def _safe_exec_no_timer(code: str):
    """
    Executes code in a restricted environment (no timeout here).
    Timeout should be applied by wrapping the whole evaluate step with _time_limit().
    """
    import math as _math

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "enumerate": enumerate,
        "int": int, "float": float, "str": str, "bool": bool,
        "round": round,
        "print": print,
    }

    def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math":
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Import blocked: {name}")

    safe_builtins["__import__"] = _limited_import

    ns = {
        "__builtins__": safe_builtins,
        "math": _math,
    }

    # Reduce noisy compile-time warnings from weird generated code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(code, ns, ns)

    return ns

def _extract_code(text: str) -> str:
    """
    Heuristics:
      - If code fences exist, prefer fenced block.
      - Else, start from first 'def ' if present.
      - Strip special tokens.
      - Trim trailing garbage until it compiles (best-effort).
    """
    # Cut at common special tokens
    for stopper in ["<|endoftext|>", "<|eot_id|>", "</s>"]:
        if stopper in text:
            text = text.split(stopper, 1)[0]

    # Prefer fenced code
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    # If it contains 'def', slice from first def
    i = text.find("def ")
    if i != -1:
        text = text[i:]

    text = text.strip()

    # Best-effort trimming to make it syntactically valid
    # (Useful if sampling adds junk after valid Python)
    lines = text.splitlines()
    for k in range(0, min(50, len(lines))):
        candidate = "\n".join(lines[: len(lines) - k]).strip()
        if not candidate:
            continue
        try:
            compile(candidate, "<sample>", "exec")
            return candidate
        except SyntaxError:
            continue

    return text


# -----------------------------
# Numeric handling functions
# -----------------------------

def _numbers_equal(pred, gold):
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) or isinstance(gold, float):
        return abs(float(pred) - float(gold)) <= 1e-3
    return int(pred) == int(gold)

def _to_number(x):
    """
    Normalize return values to int or float where possible.
    """
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return None
        xf = float(x)
        if abs(xf - round(xf)) < 1e-6:
            return int(round(xf))
        return xf
    if isinstance(x, str):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", x)
        if not m:
            return None
        s = m.group(0).replace(",", "")
        if s.count(".") == 1:
            f = float(s)
            if abs(f - round(f)) < 1e-6:
                return int(round(f))
            return f
        return int(s)
    # tuples/lists etc -> not supported for GSM8K scoring
    return None


if __name__ == "__main__":
    # tokenize the GSM8K test set first
    # test_gsm8k_tokenization(MASK_ID)

    # sanity check the eval loop with one tinygsm example
    ds = load_dataset("TinyGSM/TinyGSM", split = "train")
    ex = ds[0]

    q, a  = ex["question"], ex["code"]

    ns = _safe_exec( _extract_code(q + "\n" + a) , timeout_s = 1.0)
    out = ns["simple_math_problem"]()
    gold = str(_to_number(out))

    ok = evaluate_samples( a , gold)

    print("Sanity check passed: ", ok)
    print("--------------------------------")
    print("Question: ", q)
    print("--------------------------------")
    print("Code: ", a)
    print("--------------------------------")
    print("Answer: ", gold)
