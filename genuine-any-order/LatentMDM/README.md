<div align="center">

# LatentMDM

</div>

Public code release for the **LatentMDM** TinyGSM experiments from the
paper:

> **From Interface to Inference: Eliciting Any-Order Inference from Any-Order Models.**
> Seunggeun Kim\*, Jaeyeon Kim\*, Taekyun Lee\*, Yuyuan Chen\*, Yilun Du,
> Sham Kakade, Sitan Chen (\*co-first authors).
> [arXiv:2607.26504](https://arxiv.org/abs/2607.26504).

See the [repository root README](../README.md) for the paper and the companion
[`FlexMDM/`](../FlexMDM) codebase.

## Overview

This directory contains LatentMDM together with autoregressive (ARM) and
token-level MDM baselines:

| Strategy | Generation unit | Context and decoding |
| --- | --- | --- |
| ARM | Token | Causal context; fixed left-to-right decoding |
| MDM | Token | Bidirectional context; iterative token unmasking |
| LatentMDM | Newline-delimited answer segment | Bidirectional latent planning; autoregressive decoding within each segment |

## Environment

The project uses Python 3.11 and provides a Micromamba environment definition
in [`environment.yml`](environment.yml). Create and activate it with:

```bash
micromamba create -f environment.yml
micromamba activate LatentMDM
```

Experiment settings are OmegaConf YAML files under
[`yaml_files/`](yaml_files). Dot-list command-line values such as
`validation.limit=128` or `wandb.wandb=false` override the corresponding YAML
fields.

## Training

Run the following commands from this directory (`LatentMDM/`).

### Data preparation

[`data/tiny_gsm.py`](data/tiny_gsm.py) streams `TinyGSM/TinyGSM`, removes the
leading generated function docstring and redundant blank lines, and tokenizes
both a flat view and a newline-segmented view:

```bash
python data/tiny_gsm.py
```

The default call writes to `data/tiny_gsm` and uses 512-token flat sequences,
512-token prompts, up to 16 answer segments, and up to 32 tokens per segment.
Samples with an overflowing prompt, segment, or segment count are discarded by
default. Runtime depends on the network, CPU, and tokenizer throughput.

The resulting directory contains:

```text
tiny_gsm/
├── labels.bin          # flat prompt + answer sequences for ARM and MDM
├── prompt_mask.bin     # packed prompt mask shared by all strategies
├── prompt.bin          # fixed-width prompts for LatentMDM
├── split_labels.bin    # fixed-size answer segments for LatentMDM
└── meta.json           # shapes, dtypes, tokenizer, and filtering statistics
```

The preprocessor initially allocates space for the full source dataset before
truncating the files to the number of retained examples. With the default
dimensions, allow approximately 74 GB (69 GiB) of free disk space. The final
size can be smaller after overflow filtering.

Set `data.data_dir` in the three experiment YAMLs to the generated directory.
ARM and MDM use `data.dataset: tinygsm`; LatentMDM uses
`data.dataset: tinygsm_split`.

### Pretraining

The training wrappers are site-specific Slurm launchers. Before submitting
them, update the `#SBATCH` account, partition, email, environment name, cache
paths, and checkpoint path for your system. Each current wrapper appends a
`--resume` argument; remove it when starting from scratch.

```bash
sbatch scripts/train_arm.sh
sbatch scripts/train_mdm.sh
sbatch scripts/train_latentmdm.sh
```

## Sampling and Evaluation

Evaluation uses the GSM8K evaluator in
[`eval/gsm8k_eval.py`](eval/gsm8k_eval.py). A checkpoint can be evaluated on
one GPU through the unified entry point:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  eval.py --cfg yaml_files/tinygsm_latentmdm.yaml \
  --resume /path/to/checkpoint.pt \
  wandb.wandb=false validation.limit=128
```

Local convenience wrappers are provided for all three strategies:

- ARM: [`scripts/eval_arm.sh`](scripts/eval_arm.sh)
- MDM: [`scripts/eval_mdm.sh`](scripts/eval_mdm.sh)
- LatentMDM: [`scripts/eval_latentmdm.sh`](scripts/eval_latentmdm.sh)

These are shell scripts rather than Slurm launchers. Update their environment,
cache, network-interface, and checkpoint settings for your system, then run the
appropriate wrapper with `bash`, for example:

```bash
bash scripts/eval_latentmdm.sh
```

The evaluator expands list-valued sampling settings and reports GSM8K accuracy
for each configured confidence rule, number of slots or tokens unmasked per
step, and pass-at-k value. LatentMDM supports `avg_log_prob`, `min_log_prob`,
`random`, and `l2r` segment-selection confidence rules; see the sampling block
in the LatentMDM YAML for the remaining controls.

## Code Hierarchy

```text
LatentMDM/
├── train.py                   # training orchestration, EMA, logging, checkpoints
├── eval.py                    # checkpoint loading and evaluation entry point
├── sampling.py                # ARM, MDM, block-MDM, and LatentMDM samplers
├── utils.py                   # DDP setup, seeding, and validation-loss dispatch
├── data/
│   ├── __init__.py            # dataset-to-DataLoader dispatch
│   └── tiny_gsm.py            # TinyGSM preprocessing and memmap datasets
├── model/
│   ├── transformer.py         # shared transformer configuration and MDM backbone
│   ├── latent_mdm.py          # LatentMDM encoder/planner/decoder model
│   ├── factory.py             # strategy-aware model construction
│   └── ema.py                 # EMA and checkpoint serialization
├── training/
│   └── losses.py              # ARM, MDM, and LatentMDM objectives
├── eval/
│   ├── runner.py              # evaluation strategy/grid dispatch
│   └── gsm8k_eval.py          # GSM8K generation, execution, metrics, and traces
├── yaml_files/                # ARM, MDM, and LatentMDM experiment configurations
└── scripts/                   # local evaluation and Slurm training launchers
```

## License

Apache-2.0 — see the repository root [`LICENSE`](../LICENSE) and
[`NOTICE`](../NOTICE) (`model/ema.py` is from
[MDLM](https://github.com/kuleshov-group/mdlm), Apache-2.0).

## Acknowledgments

This codebase builds on [PUMA](https://github.com/JaeyeonKim01/PUMA).
