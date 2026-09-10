# Shared-params configs

This cycle’s knobs live in `[configs/](../../configs/)`. `configs/shared.yaml` is the comparison contract. Method files **copy those values** under each trainer’s names. Algorithm-only knobs stay in the method YAML. Launch scripts **load** these files. W&B keys: `[metrics.md](metrics.md)`.

**This cycle:** MDLM, PUMA, FlexMDM, EditFlow on **TinyGSM** and **GSM8K**.  
**Out of cycle:** LatentMDM, APPS, TACO.

Do not change a shared knob in one method only. Remaining method-inherent differences are listed at the bottom.

---

## Layout


| Path                                      | What                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| `configs/shared.yaml`                     | Values every method must share                               |
| `configs/data/tinygsm.yaml`, `gsm8k.yaml` | UNI-D2 Hydra `data=` (symlinked from `UNI-D2/configs/data/`) |
| `configs/mdlm/`                           | UNI-D2 Lightning MDLM (`algo=mdlm`, `model=small`)           |
| `configs/flexmdm/`                        | UNI-D2 Lightning FlexMDM (`algo=flexmdm-anyorder`)           |
| `configs/puma/tinygsm.yaml`               | PUMA paper trainer (TinyGSM train; GSM8K is transfer eval)   |
| `configs/editflow/`                       | EditFlow CLI flags                                           |


Launch from the method directory. Scripts load the YAML in `[configs/](../../configs/)`; do not copy knobs into the method tree.

- UNI-D2: `UNI-D2/examples/mdlm/` and `examples/flexmdm/` (`tinygsm_shared_params.sh`, `gsm8k_shared_params.sh`) → `configs/mdlm/` / `configs/flexmdm/`
- PUMA: `PUMA/launch_tinygsm256.sh` → `configs/puma/tinygsm.yaml`
- EditFlow: `edit_flows_code/scripts/run_editflow_shared_params.sh` and `run_editflow_gsm8k_shared_params.sh` → `configs/editflow/`

---

## Shared knobs

From `configs/shared.yaml`. Same meaning on every method.


| Knob                      | Value              | Why                                                                                                 |
| ------------------------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| `length`                  | `256`              | Same packed budget. EditFlow splits 128 prompt + 128 target.                                        |
| `global_batch_size`       | `64`               | Same optimizer-step size (examples × GPUs, `grad_accum=1`).                                         |
| `eval_global_batch_size`  | `32`               | Smaller likelihood / generate batch (memory). EditFlow uses `val_max_examples: 256` as a val *cap*. |
| `num_workers`             | `4`                | Dataloader workers.                                                                                 |
| `seed`                    | `42`               | Train, val split, and sampling.                                                                     |
| `precision`               | `bf16`             | Train / val autocast. PUMA hardcodes `torch.bfloat16`.                                              |
| `grad_clip`               | `1.0`              | Global grad-norm clip.                                                                              |
| `lr`                      | `3.0e-4`           | Peak AdamW LR.                                                                                      |
| `weight_decay`            | `0.03`             | AdamW decay.                                                                                        |
| `warmup_steps`            | `500`              | Linear warmup to `lr`. After this step, LR is held if `lr_schedule: constant`.                      |
| `lr_schedule`             | `constant`         | Warmup, then **fixed** peak LR. Not cosine. `cosine` is opt-in only.                                |
| `warmup_lr_init`          | `1.0e-6`           | Unused for `constant` (warmup starts at 0). Cosine-only.                                            |
| `lr_min`                  | `0.0`              | Unused for `constant`. Cosine floor if `lr_schedule: cosine`.                                       |
| `ema`                     | `0.9999`           | EMA weights at val / test. EditFlow has no EMA (trainer).                                           |
| `hidden_size`             | `768`              | Backbone width (UNI-D2 `small`, PUMA, EditFlow).                                                    |
| `num_layers`              | `12`               | Transformer depth. Same stack height on every method.                                               |
| `num_heads`               | `12`               | Attention heads. `hidden_size` must divide evenly (`768 / 12 = 64`).                                |
| `dropout`                 | `0.1`              | Residual / attention dropout. Same rate on every method.                                            |
| `val_every`               | `2000`             | Steps between likelihood val and generate+score.                                                    |
| `ckpt_every`              | `2000`             | Periodic checkpoints.                                                                               |
| `log_every`               | `100`              | `train/` log interval.                                                                              |
| `early_stopping_patience` | `9999`             | Off: every method runs the full `max_steps` budget.                                                 |
| `resume_from_ckpt`        | `false`            | Fresh comparison runs.                                                                              |
| `task_num_samples`        | `32`               | Examples scored for `test/pass@1_k*`.                                                               |
| `sampling_steps`          | `128`              | Denoising / edit steps at test. PUMA uses stage `k`, not this count.                                |
| `pass_at_1_k`             | `[1, 2, 4, 8]`     | Tokens written **per sampling step**. Same four `test/pass@1_k`* keys.                              |
| `generate_samples`        | `true`             | Run task eval at each val interval.                                                                 |
| `task_checkpoint_monitor` | `"test/pass@1_k1"` | `best-task` checkpoint. Quote the key in YAML.                                                      |


Likelihood `best` still tracks min `val/loss`.

### W&B (all methods)


| Knob            | Value             | Why                                              |
| --------------- | ----------------- | ------------------------------------------------ |
| `wandb.project` | `edit-diffusion`  | One W&B project for this cycle.                  |
| `wandb.entity`  | `alpdk-podkopaev` | Team / user that owns the project.               |
| `wandb.group`   | `shared-params`   | Filter comparison runs. Smoke/overfit use other groups. |


Run `name` is per method and dataset, e.g. `mdlm-tinygsm-L256-shared-params`.

### Dataset blocks


| Knob            | TinyGSM   | GSM8K         | What                                                           |
| --------------- | --------- | ------------- | -------------------------------------------------------------- |
| `max_steps`     | `260000`  | `20000`       | Optimizer steps. Same budget for every method on that dataset. |
| `max_examples`  | `100000`  | (full set)    | TinyGSM train cap.                                             |
| `val_ratio`     | `0.02`    | (dataset val) | TinyGSM held-out fraction for `val/`.                          |
| `eval_task`     | `tinygsm` | `gsm8k`       | In-domain `test/pass@1_k*`.                                    |
| `transfer_task` | `gsm8k`   | —             | TinyGSM runs also log `test/gsm8k/pass@1_k*`.                  |


---

## TinyGSM vs GSM8K data (UNI-D2)

`configs/data/*.yaml`. Pack `question + "\n" + answer`, no wrap, no chunking, GPT-2 tokenizer, EOS on train and val.


| Key                                     | TinyGSM                    | GSM8K                  | What |
| --------------------------------------- | -------------------------- | ---------------------- | ---- |
| `train` / `valid`                       | `tinygsm`                  | `gsm8k`                | Dataset name Hydra loads for train and likelihood val. |
| `tokenizer_name_or_path`                | `gpt2`                     | `gpt2`                 | Tokenizer. Packed length is still 256 tokens. |
| `cache_dir`                             | `${scratch_dir}/tinygsm`   | `${scratch_dir}/gsm8k` | Tokenized cache under `DISCRETE_DIFFUSION_SCRATCH_DIR`. |
| `wrap` / `chunking` / `streaming`       | `False` / `none` / `False` | same                   | One packed question + newline + answer. No wrap, no LM chunks. |
| `insert_train_eos` / `insert_valid_eos` | `True`                     | `True`                 | Append EOS on train and val packs. |
| `max_examples`                          | `100000`                   | —                      | TinyGSM train cap. GSM8K uses the full split. |
| `val_ratio`                             | `0.02`                     | —                      | TinyGSM held-out fraction. GSM8K uses its own val split. |
| `split_seed`                            | `42`                       | —                      | TinyGSM train/val split. Same as shared `seed`. |


On HPC set `DISCRETE_DIFFUSION_SCRATCH_DIR` so caches stay in the workspace.

---

## Name mapping (shared → trainer)


| Shared                    | MDLM / FlexMDM (UNI-D2)                                  | PUMA                                                                     | EditFlow                                      |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------- |
| `length`                  | `model.length`                                           | `model.max_position`, `training.reference_length`, `training.block_size` | `max_prompt_len` + `max_target_len` (128+128) |
| `global_batch_size`       | `loader.global_batch_size`                               | `training.batch_size`                                                    | `batch_size` (`grad_accum: 1`)                |
| `eval_global_batch_size`  | `loader.eval_global_batch_size`                          | same as train batch                                                      | `val_max_examples` (cap, not batch)           |
| `num_workers`             | `loader.num_workers`                                     | `data.training.cpus`                                                     | —                                             |
| `seed`                    | `seed`                                                   | `data.seed`                                                              | `seed`                                        |
| `precision`               | `trainer.precision`                                      | hardcoded `bfloat16`                                                     | `precision`                                   |
| `grad_clip`               | `trainer.gradient_clip_val`                              | `training.max_grad_norm`                                                 | `grad_clip`                                   |
| `lr`                      | `optim.lr`                                               | `training.learning_rate`                                                 | `lr`                                          |
| `weight_decay`            | `optim.weight_decay`                                     | `training.weight_decay`                                                  | `weight_decay`                                |
| `warmup_steps`            | `lr_scheduler.num_warmup_steps`                          | `training.warmup_steps`                                                  | `warmup_steps`                                |
| `lr_schedule`             | `hydra.lr_scheduler`: `constant_warmup`                  | `training.lr_schedule`                                                   | `lr_schedule`                                 |
| `ema`                     | `training.ema`                                           | `training.ema`                                                           | not implemented                               |
| `hidden_size`             | `model.hidden_size`                                      | `model.hidden_size`                                                      | `hidden_dim`                                  |
| `num_layers`              | `model.n_blocks`                                         | `model.num_layers`                                                       | `num_layers`                                  |
| `num_heads`               | `model.n_heads`                                          | `model.num_attention_heads` and `num_kv_heads`                           | `num_heads`                                   |
| `dropout`                 | `model.dropout`                                          | `model.dropout`                                                          | `dropout`                                     |
| `val_every`               | `trainer.val_check_interval`                             | `training.eval_steps`                                                    | `val_every`                                   |
| `ckpt_every`              | `callbacks.checkpoint_every_n_steps.every_n_train_steps` | `training.save_steps`                                                    | `save_every`                                  |
| `log_every`               | `trainer.log_every_n_steps`                              | `training.logging_steps`                                                 | `log_every`                                   |
| `early_stopping_patience` | `callbacks.early_stopping.patience`                      | `training.early_stopping.patience` (`enabled: false`)                    | `patience`                                    |
| `resume_from_ckpt`        | `checkpointing.resume_from_ckpt`                         | CLI `--resume` (leave unset)                                             | `--resume` (leave unset)                      |
| `task_num_samples`        | `eval.task_num_samples`                                  | full val / GSM8K test loop                                               | `task_eval_samples`                           |
| `sampling_steps`          | `sampling.steps`                                         | n/a (stage `K`)                                                          | `task_eval_steps`                             |
| `pass_at_1_k`             | eval protocol                                            | `validation.sampling.unmasking_num`                                      | same four `k` keys at test                    |
| `max_steps`               | `trainer.max_steps`                                      | `training.max_steps`                                                     | `steps`                                       |
| `eval_task`               | `eval.task`                                              | `data.dataset`                                                           | `dataset`                                     |
| `transfer_task`           | `eval.transfer_task`                                     | GSM8K eval inside TinyGSM loop                                           | GSM8K test split at val                       |


---

## Method files

Hydra composition plus knobs that are *allowed* to differ.

### MDLM — `configs/mdlm/{tinygsm,gsm8k}.yaml`


| Key                              | Value               | What |
| -------------------------------- | ------------------- | ---- |
| `hydra.data`                     | `tinygsm` / `gsm8k` | UNI-D2 data group (`configs/data/`). |
| `hydra.algo`                     | `mdlm`              | Absorbing diffusion, SUBS, ELBO. |
| `hydra.model`                    | `small`             | DIT; width pinned to 768 / 12 / 12 / dropout 0.1. |
| `hydra.sampling`                 | `default`           | Absorbing sampler. |
| `hydra.noise`                    | `log-linear`        | MDLM noise schedule (algorithm). |
| `hydra.lr_scheduler`             | `constant_warmup`   | Hold `optim.lr` after `num_warmup_steps`. |
| `eval.generate_samples`          | `true`              | Run generate+score at each val interval (`test/pass@1_k*`). |
| `callbacks.sample_saver.enabled` | `false`             | Do not dump generated text files. W&B scalars only. |
| `trainer.num_sanity_val_steps`   | `0`                 | Skip Lightning’s pre-train dummy val. |


### FlexMDM — `configs/flexmdm/{tinygsm,gsm8k}.yaml`

Same shared skeleton as MDLM, except:


| Key           | Value              | What                         |
| ------------- | ------------------ | ---------------------------- |
| `hydra.algo`  | `flexmdm-anyorder` | Unmask + insertion.          |
| `hydra.model` | `flexmdm_anyorder` | Same 768 / 12 / 12 width.    |
| `hydra.noise` | `linear`           | Insert and unmask schedules. |


Logs `train/unmask_loss`, `train/len_loss` (and `val/`). `train/loss` is their sum.

### PUMA — `configs/puma/tinygsm.yaml`

No GSM8K *training* YAML: PUMA trains on TinyGSM and scores GSM8K as transfer. Tokenizer stays Qwen because `data/tiny_gsm_256` is already tokenized.


| Key                                 | Value                 | What |
| ----------------------------------- | --------------------- | ---- |
| `model.vocab_size`                  | `151645`              | Qwen2 vocab. Not GPT-2. |
| `model.intermediate_size`           | `2048`                | SwiGLU MLP (`8/3 × 768`). No UNI-D2 equivalent. |
| `model.causal`                      | `false`               | Bidirectional (masked diffusion, not an AR LM). |
| `training.strategy`                 | `progressive`         | PUMA pool: unmask in stages instead of a single random mask. |
| `training.K`                        | `21`                  | Starting unmask quota per stage (`stats/current_k`). |
| `training.k_schedule`               | `[k, step]` pairs     | Same ladder as the paper, steps scaled 300k → 260k. |
| `training.mode`                     | `confidence_collapse` | Reveal extra high-confidence tokens inside a stage. |
| `training.confidence_threshold`     | `0.9`                 | Probability cutoff for that collapse. |
| `data.data_dir`                     | `data/tiny_gsm_256`   | Pre-tokenized L=256 Qwen packs. |
| `data.mask_id` / `eos_id`           | `151644` / `151643`   | Qwen mask and EOS ids. |
| `validation.sampling.unmasking_num` | `[1, 2, 4, 8]`        | Tokens unmasked per step; maps to `test/pass@1_k*`. |
| `validation.sampling.confidence`    | `["top_k"]`           | Unmask the highest-probability masks (protocol k). |
| `validation.sampling.temperature`   | `0.0`                 | Greedy token at unmasked positions. |


### EditFlow — `configs/editflow/{tinygsm,gsm8k}.yaml`

Conditional edit process. Lengths are **bytes** (byte tokenizer), not GPT-2 tokens.


| Key                                 | Value         | What                         |
| ----------------------------------- | ------------- | ---------------------------- |
| `max_prompt_len` / `max_target_len` | `128` / `128` | Total 256.                   |
| `coupling`                          | `empty`       | Start from empty target.     |
| `grad_accum`                        | `1`           | Global batch = `batch_size`. |
| `t_eps`                             | `1.0e-2`      | Time-sampling floor.         |
| `val_max_examples`                  | `256`         | Likelihood val cap.          |


---

## Method-inherent (do not “fix” to match)

These stay different. Everything else should match `shared.yaml`.


| Item                                                         | Why it stays                                                                  |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| MDLM `log-linear` vs FlexMDM `linear` noise                  | Algorithm.                                                                    |
| PUMA progressive `k_schedule`                                | Algorithm. Not the same as `sampling_steps`.                                  |
| FlexMDM unmask + length losses                               | Algorithm. Rank with `val/loss`, not the extra heads.                         |
| EditFlow insert/delete/substitute + `coupling` / `t_eps`     | Algorithm.                                                                    |
| Tokenizer: GPT-2 (UNI-D2) vs Qwen (PUMA) vs bytes (EditFlow) | Data / trainer. Sequence *count* is still 256.                                |
| EditFlow has no EMA                                          | Trainer has no EMA.                                                           |
| PUMA `intermediate_size: 2048`                               | SwiGLU width; DIT has no matching knob.                                       |
| PUMA TinyGSM-only train YAML                                 | No GSM8K train YAML this cycle. TinyGSM val generate+score is GSM8K transfer. |
| PUMA task eval size                                          | Trainer scores the full val / GSM8K test loop, not `task_num_samples: 32`.    |


---

## What not to tune here

UNI-D2 still has a large Hydra tree under `UNI-D2/configs/` (`algo/`, `model/`, `noise/`, `sampling/`, `callbacks/`, …). This cycle only overrides the keys in `configs/mdlm/` and `configs/flexmdm/`. Do not change `algo` parameterization or `loss_type` unless the comparison protocol changes.