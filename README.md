# Bayesian group research

Comparison workspace for discrete diffusion language models on **GSM8K** and **TinyGSM**.

This repo is not a fork of UNI-D2. Upstream UNI-D2 lives in `UNI-D2/` as a nested checkout. Method-specific trainers sit next to it so we can log the same W&B keys and compare MDLM, PUMA, FlexMDM, and EditFlow.

**This cycle:** MDLM, PUMA, FlexMDM, EditFlow.  
**Out of cycle:** LatentMDM (do not add runs or metrics). APPS and TACO are not used.

Shared metric names, prefixes, and how to read charts: [`docs/research/metrics.md`](docs/research/metrics.md). Shared-params knobs: [`docs/research/configs.md`](docs/research/configs.md). Cluster holders and node requests: [`docs/research/hpc.md`](docs/research/hpc.md).

## Layout

```text
.
├── UNI-D2/                 # upstream UNI-D2 (MDLM + FlexMDM Lightning trainers)
├── PUMA/                   # Progressive Unmasking trainer
├── edit_flows_code/        # EditFlow trainer for GSM8K / TinyGSM
├── configs/                # shared-params used this cycle (TinyGSM / GSM8K)
├── docs/research/          # comparison protocol (metrics.md, configs.md)
├── analysis/               # local checks and dataset length scripts
└── .venv/                  # Python env for UNI-D2 (create at repo root)
```


| Path                         | Role                                                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `UNI-D2/`                    | Hydra + Lightning. MDLM (`algo=mdlm`) and FlexMDM any-order (`algo=flexmdm-anyorder`). Same top-level files as [nkalyanv99/UNI-D2](https://github.com/nkalyanv99/UNI-D2). |
| `PUMA/`                      | PUMA paper trainer. TinyGSM / GSM8K launch scripts and YAML.                                                                                                              |
| `edit_flows_code/`           | Conditional Edit Flows on GSM8K / TinyGSM.                                                                                                                                |
| `configs/`                   | Shared-params YAMLs. Launch scripts load these files (not method-local copies).                                                                                          |
| `docs/research/metrics.md`   | Required W&B keys (`train/`, `val/`, `test/`, `perf/`, `healthy/`, `stats/`).                                                                                             |
| `docs/research/configs.md`   | Shared-params knobs and per-trainer names (`configs/shared.yaml`, `mdlm/`, `flexmdm/`, `puma/`, `editflow/`).                                                            |
| `analysis/`                  | Protocol parser checks and length histograms. Not the UNI-D2 test suite.                                                                                                  |




## Metrics (short)

Every logged scalar uses one prefix. Rank methods with `val/loss` (held-out objective) and `test/pass@1_k{1,2,4,8}` (task). Do not rank by `train/loss`. `perf/` is wall-clock only.

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
python -m pip install -r edit_flows_code/requirements.txt
```

Datasets cache under `~/.cache/discrete_diffusion` unless `DISCRETE_DIFFUSION_SCRATCH_DIR` is set (needed on HPC workspaces).

## How to run

Work from the method directory. Shared-params live in [`configs/`](configs/); launch scripts pass those files through.

**MDLM / FlexMDM (UNI-D2)**

```bash
cd UNI-D2
bash examples/mdlm/tinygsm_shared_params.sh      # configs/mdlm/tinygsm.yaml
bash examples/flexmdm/gsm8k_shared_params.sh     # configs/flexmdm/gsm8k.yaml
```

**PUMA**

```bash
cd PUMA
bash launch_tinygsm256.sh                        # configs/puma/tinygsm.yaml
```

**EditFlow**

```bash
cd edit_flows_code
bash scripts/run_editflow_shared_params.sh       # configs/editflow/tinygsm.yaml
bash scripts/run_editflow_gsm8k_shared_params.sh # configs/editflow/gsm8k.yaml
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

