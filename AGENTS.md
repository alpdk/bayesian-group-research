# Agent notes

This is the Bayesian-group **comparison** repo (MDLM, PUMA, FlexMDM, EditFlow on GSM8K / TinyGSM). Not a UNI-D2 fork: upstream code is in `UNI-D2/`.

Read before cluster or training work:

- [`docs/research/hpc.md`](docs/research/hpc.md) — live node-request situation, holders, `srun --overlap`, workspaces
- [`docs/research/metrics.md`](docs/research/metrics.md) — W&B keys for this cycle
- [`README.md`](README.md) — layout and how to run each method
- [`configs/`](configs/) — shared-params used this cycle

Always `squeue --me` on both `alpha` and `capella` before allocating or attaching. Do not assume old job IDs still exist.

**NEVER free jobs.** Do not `scancel` a job, holder, or pending request. Do not release, drain, or end an allocation.

If the user says **stop**, **stop jobs**, **stop all runs**, or similar, that means **training only**: cancel the overlapping training **step** (`scancel JOBID.step`) and leave the holder running. It does **not** mean free the allocation.

Whole-job `scancel` is forbidden unless the user names the JobID **and** confirms a second time after the agent restates exactly which jobs would be cancelled. No first-pass cancel.

LatentMDM, APPS, and TACO are out of this cycle.
