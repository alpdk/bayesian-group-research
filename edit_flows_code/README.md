# Edit Flows for Code Generation

Adaptation of [edit-flows-demo](https://github.com/TheMatrixMaster/edit-flows-demo)
(an educational implementation of ["Edit Flows: Flow Matching with Edit Operations"](https://arxiv.org/abs/2506.09018),
Havasi et al.) from unconditional sine-wave modeling to **conditional code/solution
generation** on grade-school math datasets:

| dataset   | prompt   | target |
|-----------|----------|--------|
| `gsm8k`   | question | chain-of-thought answer ending in `#### <num>` ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)) |
| `tinygsm` | question | Python solution `def simple_math_problem(): ... return result` ([TinyGSM/TinyGSM](https://huggingface.co/datasets/TinyGSM/TinyGSM), 11.8M rows, streamed) |

## What changed vs. the original demo

- **Conditional generation.** `CondEditFlowsTransformer` (`model.py`) encodes the
  question as a *prompt segment* that is concatenated in front of `x_t` with its own
  segment embedding. The model attends to the prompt bidirectionally, but the edit
  heads (insert/substitute/delete rates + token distributions) are only produced for
  the generated part, so the prompt is never edited.
- **Byte-level tokenizer** (`tokenizer.py`): ids 0–255 are UTF-8 bytes, plus `BOS`
  (256) and `PAD` (257) in the model vocabulary, and `GAP` (258) which only exists in
  the aligned Z-space. This replaces the hardcoded 128-value discretization of the
  sine-wave demo; all special-token ids are passed explicitly instead of module
  constants.
- **Real data** (`data.py`): GSM8K question/answer pairs or streamed TinyGSM
  question/code pairs (the code docstring, which just repeats the question, is
  stripped by default). Examples are length-filtered in bytes.
- The flow machinery (`flow.py` couplings + cubic kappa scheduler, `align.py`
  Levenshtein alignment / Z-space utilities, Bregman-divergence loss, Euler
  sampler with adaptive step size) is ported from the demo essentially unchanged.
- The default coupling is `empty`: generation starts from a lone `BOS` token and the
  model learns to build the solution by insertions (plus corrective
  substitutions/deletions). `uniform` and `extended` couplings are also available;
  note their Levenshtein alignment is O(len²) per sample in Python and slow for long
  sequences.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

(In the UNI-D2 repo the shared `.venv` at the repo root already has everything.)

## Training

```bash
# GSM8K, ~7.5k examples
python train.py --dataset gsm8k --steps 20000 --batch-size 32 --save-dir results/gsm8k

# TinyGSM (streamed subset), Python code targets
python train.py --dataset tinygsm --max-examples 200000 --steps 50000 \
    --batch-size 32 --save-dir results/tinygsm
```

Useful flags: `--max-prompt-len` / `--max-target-len` (byte length filters, default 512),
`--coupling {empty,extended,uniform}`, `--hidden-dim/--num-layers/--num-heads`,
`--device`. Checkpoints (`model.pt`), a metrics plot (`metrics.png`) and raw metrics
(`metrics.json`) land in `--save-dir`.

Validation and early stopping run by default: every `--val-every` steps (375) the full
validation set is scored with the training loss (GSM8K: the test split, same convention
as the FlexMDM/MDLM runs; TinyGSM: a `--val-ratio` hold-out, default 2%). The best model
is kept as `model_best.pt`, and training stops after `--patience` (5) validations without
improvement (`--min-delta`). Set `--val-every 0` to disable.

Note: this is a from-scratch, demo-scale model (default ~30M params) trained on a
byte vocabulary; expect syntactically plausible solutions, not GSM8K SOTA. A GPU is
strongly recommended for real runs.

## Sampling / evaluation

Prompts are taken from the GSM8K test split (the standard evaluation for
TinyGSM-trained models too):

```bash
python sample.py --checkpoint results/gsm8k/model.pt --num-prompts 16 --num-steps 500
```

- For `gsm8k`-trained models the final `#### <num>` is extracted and compared to gold.
- For `tinygsm`-trained models add `--execute` to run the generated
  `simple_math_problem()` in a subprocess (5s timeout) and compare the returned value.
  Only use this with generations you are comfortable executing.

Generations and scores are written to `samples.json` next to the checkpoint.

## Files

- `tokenizer.py` – byte-level tokenizer with BOS/PAD/GAP special tokens
- `data.py` – GSM8K / TinyGSM loading, length filtering, batch sampling
- `flow.py` – couplings (`empty`, `extended`, `uniform`) and the cubic kappa scheduler
- `align.py` – Levenshtein alignment to Z-space, gap removal, edit-target masks
- `model.py` – prompt-conditional edit-flows transformer + checkpoint I/O
- `train.py` – training CLI (Bregman divergence loss, Eq. 23 of the paper)
- `sample.py` – Euler sampling CLI with GSM8K answer extraction / code execution
