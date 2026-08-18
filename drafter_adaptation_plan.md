# Plan: adapt an EAGLE drafter for Qwen2.5-Coder-7B-Instruct on SkyRL-SQL

## Why

Measured on this box (4x L40S, batch 64, matched `gpu_memory_utilization`):

| drafter | step-1 rollout | step-2 rollout |
|---|---|---|
| none (SD off) | 148.5 s | 153.0 s (stable, ~2% var) |
| `mit-han-lab/Qwen2.5-7B-Eagle-RL` (base lineage) | 274.5 s (1.83x slower) | crashed |
| `Tengyunw/...eagle2_v0` (instruct lineage) | 136.7 s (1.09x faster) | 296.7 s (1.94x slower) |

Drafter lineage dominates. But no off-the-shelf drafter matches both our target
*and* the architecture FastRL was built around, so we train one.

## Goal and success criteria

Produce a drafter that is **measurably better than 1.09x on step 1 and still a
speedup at step 3**. Concretely:

1. **Primary**: rollout time at step 1 < 136.7 s (beat Tengyunw), with SD-off's
   148.5 s as the break-even line.
2. **Durability**: still faster than SD-off at step 3. This is the one that
   matters — see "The real blocker" below.
3. **Guardrail**: reward distribution unchanged vs SD-off (SD is lossless; we
   measured -0.105 vs -0.103, so any drift signals a bug, not a win).

Kill criterion: if the offline acceptance length does not beat the two existing
drafters after Phase 3, stop and reconsider (EAGLE3, or go straight at the
decay problem).

## Two facts that make this cheap

1. **The architecture comes for free.** `eagle_trainer.py::_build_model` derives
   the drafter config from the *target* and sets `num_hidden_layers=1`. Pointing
   `--base_model_path` at Qwen2.5-Coder-7B-Instruct yields `kv_heads=4`,
   hidden 3584, vocab 152064, maxpos 32768 -- i.e. mit-han-lab's architecture
   (theirs was made the same way from Qwen2.5-7B), and GQA so draft KV stays
   light enough to run at `gpu_mem_util` 0.6+.
2. **Warm-start works.** Verified: mit-han-lab's `pytorch_model.bin` loads into
   that freshly-built drafter with **zero shape mismatches**, and after a
   `model.` key-prefix remap, zero missing/unexpected keys except
   `lm_head.weight` -- which `_load_base_model_weights()` already copies from
   the target. So this is domain adaptation from a good initialisation, not a
   from-scratch train.

## Phases

### Phase 0 -- isolated environment (~1 h)

eagle-train pins **torch 2.6.0 / transformers 4.51.1 / deepspeed**, against
fastrl's 2.8.0 / 4.57.1. This is the exact contamination hazard that downgraded
torch under a live run earlier, so:

- build `eagle-train/.venv` as its own environment, never `uv pip install` into
  an inherited `$VIRTUAL_ENV`
- pin the interpreter in every script (`"$PY" -m ...`), never bare `python3`
- materialise the target locally: `base_model_path` must be a **directory**
  (the trainer opens `model.safetensors.index.json` by path), so symlink the HF
  snapshot dir rather than passing a repo id

Exit: `import torch, deepspeed, transformers` in the new venv, versions correct,
fastrl venv untouched.

### Phase 1 -- build the SQL corpus (~1 h)

EAGLE learns to predict *the target's own continuations*, so the corpus must be
the target generating on our exact prompts. FastRL's README is explicit that
Eagle is prefix-sensitive.

- Generation-only pass with the same engine config over train + validation
  prompts (653 + 1034), ~3 samples each -> **~5k trajectories, ~1.4e7 tokens**
- Emit parquet with a `messages` column -- `eagle_datagen.py` accepts
  `messages` or `conversations` directly
- **Sample across RL steps, not just step 0.** We measured acceptance collapsing
  after one update; mixing rollouts from steps 0/2/4 makes the drafter robust to
  drift instead of sharp only at initialisation. Cheap hedge.
- **Mix in ~20-30% general data** (`Qinghao/eagle-mix`) as regularisation: only
  653 unique prompts means real overfitting risk.

Note the existing `output/rollouts_*.jsonl` dumps are **not** sufficient -- they
truncate text at 4000 chars and omit the prompt. Needs a full-fidelity dump.

Exit: parquet loads via `load_dataset("parquet")`, spot-check that assistant
content contains the `<think>/<sql>/<observation>/<solution>` structure.

### Phase 2 -- cache hidden states (~1 h)

`eagle_datagen.py` in `eagle2` mode saves last-layer hidden state per token
(3584 dims, ~7.2 KB/token) plus `input_ids`, `loss_mask`, `seq_len`.

- `data.max_length=8192` (our actual window; the 2048 default would truncate)
- 4 GPUs (upstream script assumes 8 + Slurm -- needs a single-node torchrun)
- **~100 GB** output; 3.9 TB free, so fine

Exit: `stats.json` written, `.pt` count matches the corpus, spot-check one file
for shape `[seq_len, 3584]`.

### Phase 3 -- warm-started training (~30-60 min)

- Add ~5 lines to `_build_model`: after `_load_base_model_weights()`, load
  mit-han-lab's checkpoint with the `model.` prefix remap, `strict=False`
- Lower LR than the from-scratch default (this is adaptation) and far fewer than
  the preset 20 epochs; the README says stop at convergence
- Checkpoint frequently so Phase 4 can pick the best rather than the last
- Trainer logs only value/prob losses -- **no acceptance metric**, so a falling
  loss is necessary but not sufficient

Exit: loss plateaus; at least 2-3 checkpoints to evaluate.

### Phase 4 -- validate (~1 h)

Two tiers, cheap first:

1. **Offline**: `scripts/bench_speculative_decoding.py` for acceptance length
   and standalone speedup. No training loop, minutes per checkpoint, so this is
   how we pick the checkpoint. Needs `CUDA_HOME=/usr/local/cuda` on PATH and
   `--attention_backend triton` (flashinfer's JIT fails to build here).
2. **In-loop**: 3-step A/B against SD-off at matched `gpu_mem_util` 0.6 (GQA
   makes 0.6 affordable again, unlike Tengyunw's MHA which forced 0.5).
   Compare to the three rows in the table above, and check the reward-
   distribution guardrail.

## The real blocker (and why this plan alone is not enough)

We measured SD-on going 136.7 -> 296.7 s across a single RL update while SD-off
stayed flat. The policy drifts one GRPO step away from the drafter and
acceptance collapses. **A better drafter raises the starting point; it does not
stop the decay.**

FastRL's answer is opportunistic drafter training, which is (a) off by default
(`speculative.train.enable_drafter_training: false`) and (b) wired only into the
single-turn batch path -- hidden-state collection lives in
`_generate_with_drafter`, called only from `_batch_level_generate_sequences`,
while multi-turn uses `_handle_engine_generate` and collects nothing.

So the follow-on work, in priority order after Phase 4:

1. Bridge hidden-state collection into `_handle_engine_generate` so the drafter
   can be trained during multi-turn rollouts
2. Or, cheaper interim: periodically re-run Phases 1-3 every N RL steps
3. Separately, fix the step-2 CUDA-graph shape crash (orthogonal to drafter
   quality; graphs captured at startup don't cover post-update shapes)

If the objective is a *sustained* speedup rather than a good first step, item 1
is higher leverage than this entire plan. The honest framing: Phases 0-4 buy a
measurable step-1 number and a drafter worth keeping; the decay bridge is what
turns it into a real result.

## Cost and risk

| | estimate |
|---|---|
| Phases 0-4 | **~4-6 h** wall-clock, ~100 GB disk |
| GPU | 4x L40S, exclusive during Phases 1, 2, 4 |

| risk | mitigation |
|---|---|
| venv contamination (has bitten us) | isolated venv, pinned interpreter, never bare `python3` |
| warm-start weaker than hoped -- mit-han-lab trained on Qwen2.5-7B **base** hidden states, ours are Coder-Instruct | it only transfers layer/head priors; budget more steps, and the kill criterion catches it |
| 653 unique prompts -> overfit | mix 20-30% general data; hold out validation prompts |
| upstream scripts assume 8 GPUs + Slurm | single-node torchrun, halve per-GPU batch if needed |
| step-2 CUDA-graph crash persists | orthogonal; evaluate step 1 first, treat as separate work |
