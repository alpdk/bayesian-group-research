# W&B metrics

Every scalar belongs to exactly one prefix: `train/`, `val/`, `test/`, `perf/`, `healthy/`, `stats/`.

W&B groups charts by the first path segment. This cycle compares **MDLM, PUMA, FlexMDM, and EditFlow**. Each logs into **train / val / test / perf / healthy**. `stats/` is method internals (PUMA `k`, EditFlow edit rates).

**LatentMDM is out of this cycle and is not a later comparison target.** Do not add LatentMDM metrics, checkpoints, or runs.

**Comparability.** Do not rank methods by `train/loss`. Use `val/nll` (likelihood), `val/loss` (held-out objective), `test/pass@1_k{1,2,4,8}` (task), `perf/` (speed), `healthy/` (numerics).

Lightning may suffix `*_step` / `*_epoch`. W&B `System/` is automatic.

---



## Categories


| Prefix     | What                                                                            |
| ---------- | ------------------------------------------------------------------------------- |
| `train/`   | Objective on the current batch, plus LR. Method-specific; not comparable.       |
| `val/`     | Held-out **likelihood / loss** on packed train-domain sequences. No generation. |
| `test/`    | **Generate then score** on a task split (GSM8K / TinyGSM).                      |
| `perf/`    | Wall-clock throughput. Never a quality ranking.                                 |
| `healthy/` | Run is numerically alive: grads.                                                |
| `stats/`   | Method internals that are not the objective: PUMA `k`, EditFlow rates.          |


---



## Required (all four models)


| Key                       | Category | What                                                                                     |
| ------------------------- | -------- | ---------------------------------------------------------------------------------------- |
| `train/loss`              | train    | Mean per-token (or per-position) training objective                                      |
| `train/lr`                | train    | Optimizer LR after the scheduler                                                         |
| `val/loss`                | val      | Held-out training objective (same reduction as `train/loss`)                             |
| `val/nll`                 | val      | Held-out nats/token. AR: true NLL. Diffusion/PUMA: ELBO. Edit Flows: alias of `val/loss` |
| `test/pass@1_k1`          | test     | pass@1 with **1** token generated per sampling step                                      |
| `test/pass@1_k2`          | test     | pass@1 with **2** tokens generated per sampling step                                     |
| `test/pass@1_k4`          | test     | pass@1 with **4** tokens generated per sampling step                                     |
| `test/pass@1_k8`          | test     | pass@1 with **8** tokens generated per sampling step                                     |
| `perf/train_step_s`       | perf     | Wall seconds for one optimizer step (includes data wait)                                 |
| `perf/train_tokens_per_s` | perf     | Tokens × world size / step wall time                                                     |
| `perf/val_step_s`         | perf     | Wall seconds for one likelihood val batch (see below)                                    |
| `perf/test_s`             | perf     | Wall seconds for one full test generate+score pass                                       |
| `healthy/grad_norm`       | healthy  | Global L2 grad norm before clip                                                          |


`k` is tokens written / unmasked / emitted **per sampling step**, not the number of denoising steps. All four models log the same four keys. There is no `test/pass@1` without a `k` suffix, and no `test/pass@1_match`.

Checkpoint `best` still tracks **min** `val/nll`. Task checkpoint `best-task` tracks **max** `test/pass@1_k1` unless a run overrides `eval.task_checkpoint_monitor`. Quote `"test/pass@1_k1"` (and the other `k` keys) in YAML.

TinyGSM training runs evaluate **in-domain TinyGSM** under the unprefixed `test/` keys (`test/pass@1_k*` = Python exec vs held-out gold code). The same val also logs **GSM8K transfer** under `test/gsm8k/` (`test/gsm8k/pass@1_k*` = PUMA-style exec vs GSM8K `####` gold).

`val/bpd` and `val/ppl` are not logged: they are monotone in `val/nll`.

---



## Coverage by model


| Model        | train/                                           | val/                                     | test/                                              | perf/                                                        | healthy/    | stats/                             |
| ------------ | ------------------------------------------------ | ---------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ | ----------- | ---------------------------------- |
| **MDLM**     | `loss`, `lr`                                     | `loss`, `nll`, `nll_answer`              | `pass@1_k1`, `pass@1_k2`, `pass@1_k4`, `pass@1_k8` | `train_step_s`, `train_tokens_per_s`, `val_step_s`, `test_s` | `grad_norm` |                                    |
| **PUMA**     | `loss`, `lr`                                     | `loss`, `nll`                            | `pass@1_k1`, `pass@1_k2`, `pass@1_k4`, `pass@1_k8` | `train_step_s`, `train_tokens_per_s`, `val_step_s`, `test_s` | `grad_norm` | `current_k`                        |
| **FlexMDM**  | `loss`, `unmask_loss`, `len_loss`, `lr`          | `loss`, `nll`, `unmask_loss`, `len_loss` | `pass@1_k1`, `pass@1_k2`, `pass@1_k4`, `pass@1_k8` | `train_step_s`, `train_tokens_per_s`, `val_step_s`, `test_s` | `grad_norm` |                                    |
| **EditFlow** | `loss`, `ins_loss`, `del_loss`, `sub_loss`, `lr` | `loss`, `nll`                            | `pass@1_k1`, `pass@1_k2`, `pass@1_k4`, `pass@1_k8` | `train_step_s`, `train_tokens_per_s`, `val_step_s`, `test_s` | `grad_norm` | `u_ins`, `u_del`, `u_sub`, `u_tot` |


---



## Method extras


| Key                                                  | Who                 | What                                                                   |
| ---------------------------------------------------- | ------------------- | ---------------------------------------------------------------------- |
| `train/unmask_loss`, `train/len_loss`                | FlexMDM             | Unmask and length terms (`train/loss` is their sum)                    |
| `train/ins_loss`, `train/del_loss`, `train/sub_loss` | EditFlow            | Insert / delete / substitute Bregman terms (`train/loss` is their sum) |
| `val/unmask_loss`, `val/len_loss`                    | FlexMDM             | Same terms on the val split                                            |
| `val/nll_answer`                                     | MDLM                | Same ELBO as `val/nll`, mean over answer tokens only (see below)       |
| `test/gsm8k/pass@1_k1` … `test/gsm8k/pass@1_k8`      | TinyGSM UNI-D2 runs | GSM8K transfer at the same four tokens-per-step settings               |
| `stats/current_k`                                    | PUMA                | Tokens unmasked per progressive stage                                  |
| `stats/u_ins`, `u_del`, `u_sub`, `u_tot`             | EditFlow            | Mean insert / delete / substitute / total edit rate                    |


Paper FlexMDM (`genuine-any-order/FlexMDM`) may also log `train/insertion_loss`, `val/insertion_loss`, `train/insertion_lr` under the same prefixes. EditFlow also logs `stats/u_con`.

### `val/nll_answer`

Packed sequence is `question + "\n" + answer`. `val/nll` averages the ELBO on all non-pad tokens; `val/nll_answer` averages only the answer. The model still sees the question.

Example: `Natalia sold clips … ? \n #### 72`

- `val/nll` — mean over the whole line
- `val/nll_answer` — mean over `#### 72` only

Teacher-forced gold, not generation. Lightning UNI-D² only.

### `perf/val_step_s`

Wall seconds for **one likelihood val batch**: the same `_loss` / ELBO pass that produces `val/loss` and `val/nll`. It is **not** generation and is not `perf/test_s`.

All four models log it. MDLM and FlexMDM already get it from UNI-D2 Lightning `PerfMonitor` (`on_validation_batch_start` → `on_validation_batch_end`). PUMA and EditFlow time their val batch the same way even if they are not Lightning: clock around the held-out objective, log `perf/val_step_s`.

Do not rank quality by this key. Use it to compare val-pass cost at a given batch size, next to `perf/train_step_s` (optimizer step) and `perf/test_s` (generate+score).

---



## Aliases from older logs


| Old key                                                        | New key                                            |
| -------------------------------------------------------------- | -------------------------------------------------- |
| `loss` (PUMA, unprefixed)                                      | `train/loss`                                       |
| `val_loss` (PUMA)                                              | `val/nll`                                          |
| `val/bpd`, `val/ppl`                                           | dropped (same ranking as `val/nll`)                |
| `ema_val_acc_top_k_unmasking_2` (PUMA)                         | `test/pass@1_k1`                                   |
| `test/exec_acc`, `test/exec_acc_k3`                            | `test/pass@1_k1`, `test/pass@1_k8` (nearest old k) |
| `test/pass@1`, `test/pass@1_k3`                                | `test/pass@1_k1`, `test/pass@1_k8` (nearest old k) |
| `grad_norm` (unprefixed)                                       | `healthy/grad_norm`                                |
| `healthy/lr`                                                   | `train/lr`                                         |
| `train/gpu_mem_alloc_gb`, `healthy/gpu_mem_alloc_gb`           | dropped (W&B `System/` still has GPU mem)          |
| `train/step_s`, `train/tokens_per_s`                           | `perf/train_step_s`, `perf/train_tokens_per_s`     |
| `val/step_s`, `val/tokens_per_s`                               | `perf/val_step_s`, `perf/val_tokens_per_s`         |
| `val/task_eval_s`                                              | `perf/test_s`                                      |
| `test/exact_match`, `test/pass@1_match`                        | dropped (test is pass@1 only)                      |
| `val/exact_match`, `val/exec_acc`, `val/answer_extracted_frac` | dropped                                            |
| `test/answer_extracted_frac`, `test/exec_error_frac`           | dropped                                            |
| `val/loss` (Edit Flows)                                        | keep as `val/loss`; `val/nll` remains the alias    |
| `train/u_*`, `healthy/u_*`                                     | `stats/u_*`                                        |
| `healthy/current_k`                                            | `stats/current_k`                                  |


---



## How to read charts

1. **Likelihood (OWT / packed GSM8K / TinyGSM).** `val/nll` vs step. Within a method, also `val/loss`.
2. **Task (Python-exec, train domain).** `test/pass@1_k1` … `test/pass@1_k8`. On TinyGSM this is held-out code exec. Compare methods at the same `k`.
3. **GSM8K transfer (TinyGSM runs).** `test/gsm8k/pass@1_k1` … `test/gsm8k/pass@1_k8`.
4. **Speed.** `perf/train_tokens_per_s` (train), `perf/val_step_s` (likelihood val batch), `perf/test_s` (generate+score). Never a quality ranking.
5. **Health.** `healthy/grad_norm`.
6. **Internals.** `train/lr`. Edit Flows: `stats/u_tot` collapsing to 0 is a dead sampler. PUMA: `stats/current_k` is the progressive unmask stage.

