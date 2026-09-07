"""Token-length distributions for GSM8K (openai/gsm8k, config "main").

For each tokenizer we tokenize the question (input), the answer (output), and
the concatenation question + "\n" + answer (total, matching PUMA's packing
format), then report histograms, CDFs and a coverage table:
"fraction of examples with token length <= L".

Run:  .venv/bin/python analysis/gsm8k_length_distribution.py
Outputs are written to analysis/outputs/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
TOKENIZERS = {
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",  # tokenizer used by PUMA for TinyGSM
    "gpt2": "gpt2",
}
THRESHOLDS = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
PERCENTILES = [50, 90, 95, 99, 100]


def token_lengths(tok, texts):
    ids = tok(list(texts), add_special_tokens=False).input_ids
    return np.array([len(x) for x in ids])


def coverage_table(lengths, thresholds):
    n = len(lengths)
    return {t: (lengths <= t).sum() / n for t in thresholds}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    questions = list(ds["train"]["question"]) + list(ds["test"]["question"])
    answers = list(ds["train"]["answer"]) + list(ds["test"]["answer"])
    n = len(questions)
    print(f"GSM8K examples: {n} (train {len(ds['train'])} + test {len(ds['test'])})")

    csv_lines = ["tokenizer,part,threshold,count,fraction"]
    stat_lines = ["tokenizer,part,mean," + ",".join(f"p{p}" for p in PERCENTILES)]

    for tok_label, tok_name in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        parts = {
            "input (question)": token_lengths(tok, questions),
            "output (answer)": token_lengths(tok, answers),
            "total (q+\\n+a)": token_lengths(
                tok, [q + "\n" + a for q, a in zip(questions, answers)]
            ),
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for part_label, lens in parts.items():
            pcts = np.percentile(lens, PERCENTILES).astype(int)
            stat_lines.append(
                f"{tok_label},{part_label},{lens.mean():.1f}," + ",".join(map(str, pcts))
            )
            print(f"\n[{tok_label}] {part_label}: mean={lens.mean():.1f}, "
                  + ", ".join(f"p{p}={v}" for p, v in zip(PERCENTILES, pcts)))
            for t, frac in coverage_table(lens, THRESHOLDS).items():
                csv_lines.append(f"{tok_label},{part_label},{t},{int(frac * n)},{frac:.4f}")

            axes[0].hist(lens, bins=60, alpha=0.5, label=part_label)
            xs = np.sort(lens)
            ys = np.arange(1, n + 1) / n
            axes[1].plot(xs, ys, label=part_label, linewidth=2)

        axes[0].set_title(f"GSM8K token-length histogram ({tok_label})")
        axes[0].set_xlabel("tokens")
        axes[0].set_ylabel("examples")
        axes[0].legend()
        axes[1].set_title(f"CDF: fraction of examples with length \u2264 L ({tok_label})")
        axes[1].set_xlabel("L (tokens)")
        axes[1].set_ylabel("fraction of examples")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yticks(np.arange(0, 1.05, 0.1))
        axes[1].legend()
        fig.tight_layout()
        fig_path = os.path.join(OUT_DIR, f"gsm8k_lengths_{tok_label}.png")
        fig.savefig(fig_path, dpi=150)
        print(f"\nSaved plot: {fig_path}")

    with open(os.path.join(OUT_DIR, "gsm8k_length_coverage.csv"), "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    with open(os.path.join(OUT_DIR, "gsm8k_length_stats.csv"), "w") as f:
        f.write("\n".join(stat_lines) + "\n")
    print("Saved CSVs: gsm8k_length_coverage.csv, gsm8k_length_stats.csv")


if __name__ == "__main__":
    main()
