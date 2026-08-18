# Adapting an EAGLE drafter for Qwen2.5-Coder-7B-Instruct on SkyRL-SQL

Executed 2026-08-15/16 against `drafter_adaptation_plan.md`. Everything below was
run on 4x L40S (44.4 GiB); Phases 0-3 used only 2 GPUs, so they coexisted with
another user on the other two.

## Headline

**The drafter works: ~2.9 accepted tokens per verify vs ~1.18 for both
off-the-shelf drafters, a 2.5x improvement.** In-loop at the real 256x5 batch it
is roughly *neutral* rather than a win, because FastRL's adaptive SD disengages
above batch 32 -- but it removes the large penalty a mismatched drafter imposes.

## Offline acceptance (64 held-out prompts, max_bs 8, 8 steps / top-k 4 / 48 draft tokens)

| drafter | accept length | TPS (2 runs) |
|---|---|---|
| SD off | 1.000 | 311 / 237 |
| `mit-han-lab/Qwen2.5-7B-Eagle-RL` | 1.178 | 129 / 130 |
| `Tengyunw/qwen_2.5_7b_instruct_eagle2_v0` | 1.191 | 134 / 134 |
| **ours, warm-started, 3 epochs** | **2.873 / 2.904** | 281 / 324 |
| **ours, cold, 3 epochs** | **2.938 / 2.879** | 328 / 286 |

Accept length is the trustworthy column. TPS is noisy -- SD-off alone varied
311 vs 237 across two identical runs.

## In-loop, 256x5, gpu_mem_util 0.6 (matched)

| config | gen s1 | gen s2 | update_actor s1/s2 | step s1/s2 |
|---|---|---|---|---|
| SD off @0.6 | 145.6 | - | 942 | 1399 / - |
| SD off @0.7 | 207.8 | 186.1 | 961 / 912 | 1486 / 1388 |
| SD on, ours @0.6 | 202.6 | 183.4 | 955 / 1227 | 1469 / 1702 |
| SD on, Tengyunw @0.5 | 402.4 | 361.7 | 954 / 1280 | 1668 / 1935 |

Read carefully: our drafter's gen (202.6 / 183.4) is indistinguishable from
SD-off at 0.7 (207.8 / 186.1). Tengyunw's 402s *is* clearly worse. So the
adapted drafter removes SD's penalty without turning it into a gain.
`update_actor` swings 912-1227s across identical configs, so step totals at
n=1-2 cannot separate these; do not read the 1469 vs 1399 gap as signal.

**No decay across an RL update**: gen went 202.6 -> 183.4, i.e. it did not
degrade. The earlier Tengyunw run at batch 64 went 136.7 -> 296.7 across one
update. One update, one seed -- suggestive, not settled.

## The warm start bought nothing

The plan's "two facts that make this cheap" claimed warm-starting turns this into
domain adaptation rather than a from-scratch train. Measured:

| arm | final draft acc | final loss |
|---|---|---|
| warm (mit-han-lab init) | 86.81% | 0.076 |
| cold (random init) | **86.91%** | 0.076 |

Identical. The warm run even *started* at 7.5% accuracy -- cold-start territory.
Architectural compatibility (verified: 0 shape mismatches, 0 unexpected keys,
only `lm_head.weight` missing) is **not** functional transfer: mit-han-lab
trained against Qwen2.5-7B **base** hidden states, ours come from
Coder-7B-**Instruct**.

This is fine in practice -- an epoch is ~6 minutes, so the whole training phase
is cheap either way. But the plan's load-bearing fact is the *other* one: the
architecture (hidden 3584, kv_heads 4 GQA, vocab 152064, maxpos 32768) comes free
from the target config, and GQA is why our drafter fits at `gpu_mem_util` 0.6
where Tengyunw's MHA (kv_heads 28, ~7x the draft KV) OOMs.

## Actual cost vs plan estimate

| phase | plan | actual |
|---|---|---|
| 0 venv | ~1 h | ~20 min |
| 1 corpus (5061 trajectories) | ~1 h | ~20 min, 0 failures |
| 2 hidden states | ~1 h, ~100 GB | ~8 min, **70 GB** |
| 3 training (3 epochs) | 30-60 min | ~18 min/arm |
| 4 validation | ~1 h | ~30 min offline + ~50 min in-loop |

## Reproduce

```bash
# 0. isolated venv (torch 2.6 / transformers 4.51 -- NOT fastrl's or granular's)
bash eagle-train/install_uv.sh

# 1. corpus (fastrl venv: needs sglang + skyrl_gym)
bash examples/sql/gen_corpus.sh

# 2. hidden states (eagle-train venv, 2 GPUs)
bash eagle-train/scripts/datagen_sql_all.sh

# 3. train warm + cold
bash eagle-train/scripts/train_sql_ab.sh

# 4. export + benchmark
python eagle-train/scripts/export_drafter.py \
    --ckpt /local_nvme1/mborjigi/eagle-ckpt/sql-coder7b-cold --tag global_step2793 \
    --target eagle-train/models/qwen2.5-coder-7b-instruct-target \
    --out /local_nvme1/mborjigi/eagle-drafters/sql-cold-e3
bash scripts/bench_drafters.sh
bash examples/sql/run_inloop_ab.sh
```

Artifacts: drafters in `/local_nvme1/mborjigi/eagle-drafters/{sql-warm-e3,sql-cold-e3}`,
DeepSpeed checkpoints (3 epochs each arm) in `/local_nvme1/mborjigi/eagle-ckpt/`,
hidden-state cache in `/local_nvme1/mborjigi/eagle-cache/` (70 GB, deletable).

## Upstream landmines fixed (none documented)

1. **flash-attn ABI**: the README's `cxx11abiTRUE` wheel is wrong for PyPI torch
   2.6, which is built `cxx11abiFALSE` (that flipped in 2.7). Symptom:
   `undefined symbol: _ZN3c105ErrorC2E...`, which also takes deepspeed down.
2. **`setuptools` is not seeded by `uv venv`** but triton's nvidia backend imports
   it at load; without it `import deepspeed` fails and every op reports
   "compatibility check failed".
3. **transformers <-> deepspeed circular import**: `import deepspeed` must precede
   `transformers`. Patched at the top of `eagle_datagen.py`.
4. **`hydra` missing from `requirements.txt`** -- `eagle_datagen.py` imports it.
5. **Model type is detected by substring-matching the model PATH**, not the
   config. A directory not containing "qwen" silently takes the generic branch,
   leaves `model_type=None`, and dies later with `Unsupported model type: None`.
6. **`dataset_max_len` is parsed from a SUBDIRECTORY name** of `--data_path`
   (split on `-`, field [1], e.g. `8K` -> 8192). `.pt` files at the top level
   mean no subdirectories, so it keeps its 2048 default and the collator computes
   a negative pad width: `Trying to create tensor with negative dimension -778`.
   Hence the mandatory `data-8K/` layout.
7. **Stock deepspeed config is sized for a different run**: `warmup_num_steps`
   12000 and `total_num_steps` 800000, against our ~931 steps/epoch. The LR would
   sit near zero for ~13 epochs and the run would look inert. Ours: 50 / 1500.
8. **The collator pads every batch to the GLOBAL max**, not the batch max, so the
   lm_head materialises `B x 8192 x 152064` logits every step regardless of actual
   length -- 9.28 GB at B=2, which OOMs a 44 GB card. Micro-batch must be 1.
   (Fixing the collator to pad per-batch would be a real speedup, but it changes
   the loss normalisation, which divides by the padded length.)

## Known limitations

* **Observation tokens are supervised.** Our protocol is one continuous assistant
  turn, so environment-injected `<observation>` blocks fall inside the assistant
  span and get `loss_mask=1`. The drafter is trained to predict tokens it never
  drafts at inference. Mild impurity; fixing it means deriving the mask from the
  interaction's turn boundaries rather than the chat template.
* **Step-0 policy only.** The plan wanted trajectories sampled across RL steps
  0/2/4 to harden against drift. No RL checkpoints exist (`save_freq=-1`
  throughout), so this corpus is all from the initial policy. That is the arm
  that would address decay.
* **The batch-size gate is untouched.** At 256x5 (~1280 concurrent requests) SD
  is off for most of the rollout by construction. The obvious next experiment is
  a `speculative.bs_threshold` sweep now that a drafter with 2.9 accept length
  exists -- previously that would have been testing a bad drafter.
