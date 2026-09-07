# Bayesian group research

Comparison workspace for discrete diffusion language models on **GSM8K** and **TinyGSM**.

This repo is not a fork of UNI-D2. Upstream UNI-D2 lives in `UNI-D2/` as a nested checkout. Method-specific trainers sit next to it so we can log the same W&B keys and compare MDLM, PUMA, FlexMDM, and EditFlow.

**This cycle:** MDLM, PUMA, FlexMDM, EditFlow.  
**Out of cycle:** LatentMDM (do not add runs or metrics). APPS and TACO are not used.

Shared metric names, prefixes, and how to read charts: [`docs/research/metrics.md`](docs/research/metrics.md). Cluster holders and node requests: [`docs/research/hpc.md`](docs/research/hpc.md).

## Layout

```text
.
├── UNI-D2/                 # upstream UNI-D2 (MDLM + FlexMDM Lightning trainers)
├── PUMA/                   # Progressive Unmasking trainer
├── genuine-any-order/      # paper FlexMDM (and LatentMDM, unused this cycle)
│   └── FlexMDM/
├── edit_flows_code/        # EditFlow trainer for GSM8K / TinyGSM
├── configs/                # shared-params used this cycle (TinyGSM / GSM8K)
├── docs/research/          # comparison protocol (metrics.md)
├── analysis/               # local checks and dataset length scripts
└── .venv/                  # Python env for UNI-D2 (create at repo root)
```


| Path                         | Role                                                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `UNI-D2/`                    | Hydra + Lightning. MDLM (`algo=mdlm`) and FlexMDM any-order (`algo=flexmdm-anyorder`). Same top-level files as [nkalyanv99/UNI-D2](https://github.com/nkalyanv99/UNI-D2). |
| `PUMA/`                      | PUMA paper trainer. TinyGSM / GSM8K launch scripts and YAML.                                                                                                              |
| `genuine-any-order/FlexMDM/` | Paper FlexMDM (Dream-Coder insertion + unmask). Use this when the Lightning FlexMDM path is not the paper code.                                                           |
| `edit_flows_code/`           | Conditional Edit Flows on GSM8K / TinyGSM.                                                                                                                                |
| `configs/`                   | Experiment YAMLs for this cycle (`shared.yaml`, then `mdlm/`, `flexmdm/`, `puma/`, `editflow/`). Launch scripts in each method dir still apply these knobs.               |
| `docs/research/metrics.md`   | Required W&B keys (`train/`, `val/`, `test/`, `perf/`, `healthy/`, `stats/`).                                                                                             |
| `analysis/`                  | Protocol parser checks and length histograms. Not the UNI-D2 test suite.                                                                                                  |




## Metrics (short)

Every logged scalar uses one prefix. Rank methods with `val/nll` (likelihood), `val/loss` (held-out objective), and `test/pass@1_k{1,2,4,8}` (task). Do not rank by `train/loss`. `perf/` is wall-clock only.

`k` is **tokens generated per sampling step** (1, 2, 4, 8), not the number of denoising steps. TinyGSM also logs GSM8K transfer under `test/gsm8k/pass@1_k`*.

Details: `[docs/research/metrics.md](docs/research/metrics.md)`.

## Setup

Python env for UNI-D2 (repo root `.venv`):

```bash
cd /path/to/bayesian-group-research
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e UNI-D2
```

Optional CUDA extras after the editable install: `flash-attn`, `liger-kernel`. See `UNI-D2/README.md`.

Other trainers use their own environments:

```bash
conda env create -f PUMA/environment.yml          # conda env: puma
# FlexMDM paper code: genuine-any-order/FlexMDM/environment.yml
python -m pip install -r edit_flows_code/requirements.txt
```

Datasets cache under `~/.cache/discrete_diffusion` unless `DISCRETE_DIFFUSION_SCRATCH_DIR` is set (needed on HPC workspaces).

## How to run

Work from the method directory. Launch scripts and Hydra overrides live there; this README does not duplicate them.

**MDLM / FlexMDM (UNI-D2)**

```bash
cd UNI-D2
PYTHONPATH=src python -u -m discrete_diffusion \
  data=tinygsm \
  algo=mdlm \
  # ... see examples/mdlm/tinygsm_shared_params.sh
```

Shared-param examples: `UNI-D2/examples/mdlm/`, `UNI-D2/examples/flexmdm/` (`tinygsm_shared_params.sh`, `gsm8k_shared_params.sh`).

**PUMA**

```bash
cd PUMA
# e.g. launch_tinygsm256.sh  — see PUMA/README.md
```

**Paper FlexMDM**

```bash
cd genuine-any-order/FlexMDM
# see README.md and scripts/
```

**EditFlow**

```bash
cd edit_flows_code
# scripts/run_editflow_gsm8k_shared_params.sh
# scripts/run_editflow_shared_params.sh   # TinyGSM
```

Each trainer must log the keys in `docs/research/metrics.md`. Lightning UNI-D2 already logs `perf/val_step_s` via `PerfMonitor`. PUMA and EditFlow should time a val batch the same way.

## Analysis

```bash
# protocol parsers (expects UNI-D2 eval modules on PYTHONPATH)
PYTHONPATH=UNI-D2/src python analysis/test_eval_protocol.py
```

`analysis/gsm8k_length_distribution.py` and `analysis/tinygsm_length_distribution.py` print token-length stats for packing.

## Upstream

- UNI-D2: [nkalyanv99/UNI-D2](https://github.com/nkalyanv99/UNI-D2)
- PUMA: [arXiv:2602.10314](https://arxiv.org/abs/2602.10314)
- FlexMDM: [arXiv:2607.26504](https://arxiv.org/abs/2607.26504)
- Edit Flows: [arXiv:2506.09018](https://arxiv.org/abs/2506.09018)

