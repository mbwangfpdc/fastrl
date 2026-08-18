# SkyRL-SQL workload on FastRL

Runs the SkyRL-SQL multi-turn text-to-SQL task on FastRL/verl so the adaptive
speculative decoding speedup can be measured on it.

## Design: an Interaction, not a Tool

SkyRL-SQL is a **raw-tag** protocol — the policy emits `<sql>...</sql>`, the env
replies `<observation>...</observation>`, the episode ends at
`<solution>...</solution>`. It is *not* OpenAI-style function calling.

verl's tool path gates on `self._function_call_parser.has_tool_call(content)`
(`sglang_rollout.py:1140`), which expects JSON tool calls. Registering tools
would inject tool schemas into the chat template and require the model to emit
JSON — changing prompt shape, turn structure and response-length distribution,
which is exactly the long-tail profile FastRL's SD is being measured against.

verl reaches the `INTERACTING` state precisely when *no* tool call was parsed
(`sglang_rollout.py:1180-1187`), which is the seam that leaves the protocol
untouched. So the loop is implemented as a `BaseInteraction`:

    verl/interactions/sql_interaction.py  ->  skyrl_gym.envs.sql.SQLEnv

Delegating to `skyrl_gym` means observation formatting (including the
`<reminder>` suffix and 50-row truncation), termination and reward are the same
code paths the published SkyRL-SQL numbers came from. `skyrl_gym` imports
nothing from `skyrl_train`, so this pulls in no rollout/engine machinery —
`func-timeout, omegaconf, pandas, requests` only.

Adaptive SD needs nothing here: it lives inside sglang
(`third-party/sglang/.../speculative/eagle_mab.py`) and is configured via engine
server args, so any generation path through the engine gets it.

## Files

| file | role |
|---|---|
| `verl/interactions/sql_interaction.py` | the multi-turn loop (wraps `SQLEnv`) |
| `examples/sql/interaction_config.yaml` | registers it as interaction `sql` |
| `examples/sql/sql_reward.py` | training reward (`custom_reward_function`) |
| `examples/sql/prepare_skyrl_sql_data.py` | dataset → verl format |
| `examples/sql/test_sql_interaction.py` | CPU-only checks, no GPU/model |
| `examples/grpo_7B_sql.sh` | end-to-end run |

## Quick start

```bash
# 0. one-time dep
uv pip install skyrl-gym==0.3.0          # with VIRTUAL_ENV=<fastrl>/.venv

# 1. data
python examples/sql/prepare_skyrl_sql_data.py \
    --input  /local_nvme1/mborjigi/data/text2sql-data/train.parquet \
    --output data/skyrl_sql/train.parquet \
    --db-path /local_nvme1/mborjigi/data/text2sql-data/data --split train
python examples/sql/prepare_skyrl_sql_data.py \
    --input  /local_nvme1/mborjigi/data/text2sql-data/validation.parquet \
    --output data/skyrl_sql/validation.parquet \
    --db-path /local_nvme1/mborjigi/data/text2sql-data/data --split validation

# 2. verify the loop without a GPU
python examples/sql/test_sql_interaction.py

# 3. train / benchmark  (SD on, then the A/B baseline)
bash examples/grpo_7B_sql.sh
SPECULATIVE=false bash examples/grpo_7B_sql.sh
```

## Two rewards, on purpose

The interaction returns a per-turn reward, but verl's rollout stores that in
`reward_scores` and **no reward manager reads it back**. The training signal is
`examples/sql/sql_reward.py` scoring the decoded response. That split is useful:
a trajectory cut short by `max_assistant_turns` (so the env never hit its
terminal branch) still gets scored correctly.

Both use skyrl_gym's `calculate_reward_single`: `-1.0` format error, `1.0` gold
match, `0.0` otherwise. This is the strict column-order-sensitive training
reward — a few points below the OmniSQL benchmark metric, so don't compare it
directly to published exec-acc.

## Settings that matter for fidelity

- **`stop=['</sql>','</solution>']` with `no_stop_trim=True`.** Non-negotiable.
  Without stop strings the policy runs past `</sql>` and hallucinates its own
  observations; without `no_stop_trim` the closing tag is trimmed and
  `SQLEnv._parse_action`'s `<sql>(.*?)</sql>` regex never matches. `verl`'s
  `_init_sampling_params` forwards any config key containing `"stop"`, hence the
  `+` overrides in the run script.
- **Leave `multi_turn.tool_config_path` unset** so no function-call parser is
  installed.
- **`max_assistant_turns` > the env's `max_turns`** so the environment, not the
  rollout loop, decides when an episode ends.
- Sampling mirrors granular-cais-rl's SQL config: `temperature=0.6`,
  `top_p=0.95`, `n=5`, `max_prompt_length=4096`, 5 assistant turns.

## Known gaps

- **Per-turn generation cap.** granular-cais-rl caps each turn at 3000 tokens;
  verl's `response_length` is a single knob used both as the per-call
  `max_new_tokens` and as the size of the response tensor, so it is set to the
  whole-episode budget (12288) instead. Long individual turns are therefore
  possible here but not there, which shifts the tail slightly.
- **Drafter/target match.** EAGLE drafters are trained per target model and are
  prefix-sensitive. `mit-han-lab/Qwen2.5-7B-Eagle-RL` targets `Qwen2.5-7B`; using
  it with `NovaSky-AI/SkyRL-SQL-7B` will accept poorly and understate the
  speedup. For a headline number train a drafter for the SQL policy with
  `eagle-train/`.
- **Opportunistic drafter training is single-turn only.** Hidden-state
  collection lives in `_generate_with_drafter`, whose only caller is
  `_batch_level_generate_sequences` (`sglang_rollout.py:753`); the multi-turn
  path uses `_handle_engine_generate` and collects nothing. It is off by default
  (`speculative.train.enable_drafter_training: false`), so this does not affect
  a plain SD benchmark — but `enable_drafter_training=true` will silently
  train no drafter under `multi_turn.enable=true` until that hook is mirrored.
- **The rollout loop has not been exercised against a live engine.** The box's
  GPUs were held by another user throughout. See "Verification status" below.

## GPU validation results (2026-08-14)

Config mirrors granular-cais-rl `sql_baseline.toml` (`examples/grpo_sql_baseline.sh`)
from **Qwen/Qwen2.5-Coder-7B-Instruct**, batch 64 x 5 generations on 4x L40S.

### It learns: 10 steps, no SD, zero errors

`output/train10.log` (batch 64 x 5, `SPECULATIVE=false`, ~450s/step, 0 tracebacks):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean score | -0.078 | -0.231 | -0.034 | -0.097 | +0.091 | +0.184 | +0.216 | +0.256 | +0.256 | **+0.506** |
| grad_norm | 0.165 | 0.139 | 0.173 | 0.153 | 0.136 | 0.137 | 0.137 | 0.137 | 0.144 | 0.117 |
| response_length | 759 | 856 | 735 | 762 | 766 | 768 | 701 | 717 | 661 | 677 |

Decomposed from the rollout dump (`output/rollouts_train.jsonl`, 320/step):

| step | correct | wrong | format error | has `<solution>` | `<sql>` turns |
|---|---|---|---|---|---|
| 1 | 34.4% | 23.4% | 42.2% | 70.9% | 3.09 |
| 3 | 37.2% | 22.2% | 40.6% | 74.4% | 2.94 |
| 5 | 42.2% | 24.7% | 33.1% | 73.4% | 2.95 |
| 6 | **49.1%** | 20.3% | **30.6%** | 74.1% | 3.00 |

Both terms move the right way -- the policy learns the protocol (format errors
42.2% -> 30.6%) *and* the task (correct 34.4% -> 49.1%) -- while turn count stays
flat and responses get shorter (856 -> 677 tokens). grad_norm is stable
throughout, well under the 0.5 clip.

### Training signal matches granular-cais-rl

| metric | this harness | granular `big_rec` |
|---|---|---|
| response_length mean | 710 - 824 | 823 |
| frac correct (reward 1.0) | 33 - 37% | 34.8% |
| frac format error (-1.0) | 41 - 46% | 34.2% |
| mean score | -0.06 to -0.24 | +0.006 |
| multi-turn rollouts | 99.8% | n/a |
| mean `<sql>` turns | 3.2 of 5 | n/a |

Rollouts were inspected directly (`output/rollouts_*.jsonl`): observations are
injected with the `<reminder>` suffix, `n_sql` spans the full 1-5 turn budget,
and the format errors are genuine model behaviour (emitting `<sql>` straight
after `</observation>` without the mandated `<think>`, or burning all 5 turns
debugging SQL errors) -- not harness artifacts.

### Drafter choice matters more than anything else

There is **no EAGLE drafter on HuggingFace for Qwen2.5-Coder-7B-Instruct**. But
Qwen2.5-Coder-7B-Instruct is architecturally identical to Qwen2.5-7B-Instruct
(`Qwen2ForCausalLM`, 28 layers, hidden 3584, vocab 152064, kv_heads 4), so
drafters for the latter load against it. `mit-han-lab/Qwen2.5-7B-Eagle-RL`
targets Qwen2.5-7B **base**; `Tengyunw/qwen_2.5_7b_instruct_eagle2_v0` targets
the **instruct** model -- a closer lineage -- and is the same
`Qwen2ForCausalLMEagle` class sglang's `qwen2_eagle.py` loads.

Matched A/B, `gpu_memory_utilization=0.5`, batch 64, rollout time per step:

| | step 1 gen | step 2 gen | response_length |
|---|---|---|---|
| SD off | 148.5 s | 153.0 s | 737 / 803 |
| SD on, `Tengyunw` instruct drafter | **136.7 s** (1.09x faster) | **296.7 s** (1.94x slower) | 717 / 822 |
| SD on, `mit-han-lab` base drafter (at 0.6) | 274.5 s (1.83x slower) | crashed | 723 |

Response lengths match across arms, so the workload is equivalent and the gap is
genuinely SD behaviour. Two things stand out:

1. **The instruct-lineage drafter makes SD actually pay off -- on step 1.** It
   goes from 1.83x slower (base drafter) to 1.09x *faster*. It also survived
   step 2, where the base drafter crashed in CUDA-graph replay.
2. **The gain evaporates after one RL update.** SD-off rollout time is stable
   (148.5 -> 153.0 s, ~2% variance) while SD-on jumps 136.7 -> 296.7 s. The
   policy drifts one GRPO step away from what the drafter was trained near,
   acceptance collapses, and draft/verify overhead dominates.

Point 2 is precisely what FastRL's *opportunistic drafter training* exists to
prevent -- and it is off by default (`enable_drafter_training: false`) **and**
wired only into the single-turn batch path (see the drafter-training caveat
below), so it cannot currently keep a multi-turn SQL drafter fresh.

Two handicaps remain even at step 1: the drafter is not code/SQL-domain, and it
declares a 2048 training context (overridden via
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`) while these sequences run 2800+
tokens, so acceptance likely degrades past position 2048. A SQL-domain drafter
trained at full context should beat 1.09x.

Caveat: one seed, two steps. Suggestive, not conclusive.

Repro:

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SPEC_MODEL_PATH=Tengyunw/qwen_2.5_7b_instruct_eagle2_v0 \
GPU_MEM_UTIL=0.5 TRAIN_PROMPT_BSZ=64 MINI_BSZ=64 MAX_STEPS=2 SPECULATIVE=true \
bash examples/grpo_sql_baseline.sh
```

`GPU_MEM_UTIL` must drop to 0.5 for this drafter: it uses full MHA
(`kv_heads=28`) versus mit-han-lab's GQA (4), so its draft-time KV is ~7x
heavier and the engine OOMs at 0.6.

### Speculative decoding A/B, original base drafter (matched settings, step 1)

| | gen (s) | step (s) | mean score | mean turns |
|---|---|---|---|---|
| SD off | 150.2 | 536.6 | -0.105 | 3.20 |
| SD on  | 274.5 | 654.6 | -0.103 | 3.08 |

**SD is ~1.8x slower here, and that is expected**: the only available EAGLE
drafter (`mit-han-lab/Qwen2.5-7B-Eagle-RL`) is trained for `Qwen2.5-7B`, not the
`Qwen2.5-Coder-7B-Instruct` policy, so acceptance is poor and the draft/verify
overhead dominates. A drafter trained for this target (via `eagle-train/`) is a
prerequisite for a meaningful speedup number.

Encouragingly, the reward distributions are statistically indistinguishable
(mean score -0.105 vs -0.103, turns 3.20 vs 3.08), which is what lossless SD
should produce -- speed changes, distribution does not.

**SD is not yet stable across steps.** It completes step 1 then dies on step 2
in sglang's CUDA-graph replay (`size of tensor a (10) must match tensor b
(1106)`): the graphs captured at startup do not cover the shapes seen after a
weight update. SD-off runs 2+ steps cleanly.

### Fixes required to get here

All are in-tree; each was a real blocker.

| fix | why |
|---|---|
| `use_conversation_multi_turn=False` (new; `schemas.py`) | **Correctness.** verl's chat-template multi-turn interposes the literal role word, so the reward parser -- which requires `<think>` right after every `</observation>` -- scored **every** multi-turn trajectory -1.0. Silent. |
| `_get_or_create_event_loop()` (`sglang_rollout.py`) | `asyncio.get_event_loop()` raises under Python 3.12 + uvloop in Ray worker threads. |
| restore `sleep()` in `fsdp_sglang.__exit__` | Upstream leaves it commented out; the engine's ~18GB stayed pinned through training and OOMed the backward on a 44GB card. |
| `enforce_eager` -> sglang `disable_cuda_graph` | The config key existed but was never passed to the engine. |
| `CUDA_HOME=/usr/local/cuda` on PATH | sglang JIT-compiles CUDA-graph kernels; without nvcc SD dies with "Could not find nvcc". nvcc is present but off PATH. |
| `attention_backend=triton` | flashinfer's JIT fails to compile here (33 template errors in `quantization.cu`). |
| Ulysses SP=2 + `ppo_max_token_len_per_gpu=4096` | verl asserts `budget * sp >= max_seq_len`, so the budget cannot be lowered alone; SP=2 halves per-GPU logits memory and clears the step-2 OOM. |
| `tokenization_sanity_check_mode=disable` | It re-renders the chat template per request and now always differs by design. Worth **35% of rollout time** (gen 182.7s -> 119.3s). |
| stop-tag sanitizer in the interaction | skyrl-gym 0.3.0 *asserts* nothing follows `</sql>`; sglang emits 1-2 trailing newlines. (Verified benign -- stop strings do reach the engine.) |

Deviations from `sql_baseline.toml`, all infrastructure rather than workload:
batch 64 (not 256) for iteration speed; `token_budget` 4096 with SP=2 (not
24000) because verl materialises full `T x 152k` logits per microbatch;
`gpu_memory_utilization` 0.6 (not 0.9) since training and inference share the
card.

## Verification status

Verified:

- `examples/sql/test_sql_interaction.py` — 17/17 against a real SynSQL SQLite
  database: observation wrapping, `<reminder>`, turn budget, terminal
  `<solution>`, reward `1.0`/`0.0`/`-1.0`, env release and eviction bound.
- Dataset round-trip: `RLHFDataset` loads the converted parquet and yields
  `interaction_kwargs` as a plain dict with all five keys, `tools_kwargs` empty.
- `initialize_interactions_from_config` resolves the YAML to
  `{'sql': SQLInteraction}`.
- Full hydra resolution of the run config, including
  `stop: [</sql>, </solution>]` and `no_stop_trim: true`.
- A partial single-GPU run got as far as `[validate_config] All configuration
  checks passed successfully!` with the custom reward function loaded and both
  dataloaders built (64 rows, 0 filtered), before being stopped.

Not verified: generation through sglang with the interaction in the loop, the
observation tokens landing in `responses` with `response_mask=0`, and the GRPO
step itself. Run the smoke test below when the GPUs are free.

A 64/16-row dataset is already prepared at `data/skyrl_sql_small/`.

```bash
# ~2 GB on one GPU, no drafter, one training step.
# NB: do NOT set HF_HUB_OFFLINE=1 -- sglang probes for an optional
# hf_quant_config.json and offline mode turns that normal 404 into a hard error.
# Pass only keys the script does not already set, or Hydra rejects the duplicate.
CUDA_VISIBLE_DEVICES=0 HF_HOME=/local_nvme0/mborjigi/hf \
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct SPECULATIVE=false \
NGPUS=1 TP=1 ULYSSES=1 TRAIN_PROMPT_BSZ=4 N_RESP=2 MINI_BSZ=2 \
MAX_RESPONSE_LENGTH=2048 DATA_PATH=data/skyrl_sql_small \
bash examples/grpo_7B_sql.sh trainer.total_training_steps=1
```

A 0.5B model will mostly emit format errors (reward `-1.0`) — that is expected
and fine; the point is that the loop turns, observations are injected, and a
step completes.

---

## Throughput tuning at the real 256x5 batch (2026-08-15)

Goal: give FastRL a *reasonable, fair* configuration on this hardware for a
system-vs-system comparison -- not the global optimum.

All numbers: Qwen2.5-Coder-7B-Instruct, 4x L40S (44.4 GiB), `train_batch_size`
256, `rollout.n` 5, everything else as in `examples/grpo_sql_baseline.sh`.
Timings are `timing_s/*` from the run logs, in `output/opt_sweep/`.

### Result

| config | `gpu_mem_util` | SD | step 1 | step 2 |
|---|---|---|---|---|
| c0_control | 0.5 | off | 1536s | - |
| f1_tuned (+wsync 2048/wake/prefetch) | 0.5 | off | 1516s | - |
| e1_mem06 | 0.6 | off | **1399s** | - |
| g1_mem07_sdoff | 0.7 | off | 1486s | **1388s** |
| g3_mem05_sd | 0.5 | on (Tengyunw) | 1668s | 1935s |

**Fair-shot config: `gpu_mem_util` 0.6-0.7, speculative decoding OFF.** These are
now the script defaults. 0.6 and 0.7 are equivalent within noise; 0.5 costs ~10%.

### Phase decomposition (g1, step 2)

| phase | time | share |
|---|---|---|
| `update_actor` | 912s | 66% |
| `old_log_prob` | 268s | 19% |
| `gen` | 186s | 13% |

The training side dominates at this batch size, and it is close to roofline:
`old_log_prob` is a pure forward over ~3.2M tokens, implying ~600 TFLOP/s
aggregate across 4 L40S. So generation-side tricks have a small ceiling here.

### Why speculative decoding does not help at 256x5

FastRL's SD is **adaptive**: `scheduler.py` sets `spec_algorithm = NONE`
whenever the running batch exceeds `speculative.bs_threshold` (default 32), and
`eagle_worker.py` only re-enables it after 10 consecutive decode batches below
that. At 256x5 there are ~1280 concurrent requests, so speculation engages only
on the tail. Meanwhile enabling SD forces `disable_overlap_schedule=True` and
`cuda_graph_max_bs=32` engine-wide (`sglang_rollout.py`, `speculative_args`),
which taxes the dominant high-batch phase. Net: gen 269s -> 402s.

SD is *lossless* for quality (reward distributions matched in the earlier
batch-64 A/B); it is purely a throughput question, and at this batch size the
answer is negative.

### Opportunistic drafter training cannot be enabled by configuration

`speculative.train.enable_drafter_training` is off by default, and turning it on
does nothing useful in our multi-turn setup:

* Hidden-state collection has two paths. The engine-side one lives in
  `_batch_level_generate_sequences`, but `generate_sequences` routes to
  `_req_level_generate_sequences` whenever `multi_turn.enable` is set, so it is
  dead for us. The trainer-side one (`ray_trainer.py`, during `old_log_prob`)
  *is* mode-agnostic and would work.
* But the training loop is only ever started by `_check_and_start_training`,
  reachable solely from `release_worker_memory()` in `_generate_with_drafter` --
  again single-turn only.
* And `_training_step_impl` returns `False` unless
  `collect_hidden_states_from_sgl=true`, which also gates the `DataBuffer`.
* Latent bug: `worker_manager.py` initialises `current_rl_step = 0` and never
  increments it, freezing the worker-side interval check.

So enabling it allocates a drafter model + optimizer per GPU and collects data
that is never trained on. Wiring it up for multi-turn is a code change (bridge
collection into `_handle_engine_generate`, relax the `collect_*` gate), not a
config flag. The flags are plumbed (`DRAFTER_TRAINING`, `DRAFTER_COLLECT_SGL`)
but left off.

### Knobs that are NOT available on 44 GiB cards

* `enable_gradient_checkpointing=False` -- OOMs in `update_actor` at 41.4-41.9
  GiB of 44.4. The step is memory-bound into a compute-bound configuration.
* `param_offload=False` -- sglang then fails at init with
  `max_total_num_tokens <= 0`, because it sizes its KV pool from memory that is
  actually free and the resident FSDP params (fp32, ~7.6 GiB/GPU) starve it.
  Offload here is load-bearing, not a slow default.
* SD + an MHA drafter needs `gpu_mem_util` 0.5; `Tengyunw`'s drafter has
  `kv_heads=28` vs 4, so its draft KV is ~7x heavier and OOMs at 0.6.

### Caveats that may understate FastRL

* `attention_backend=triton`: `fa3` needs Hopper (these are Ada), and
  flashinfer's JIT fails to build here (33 template errors in
  `quantization.cu`). FastRL would likely look better on H100s.
* `ppo_max_token_len_per_gpu=4096` with Ulysses SP=2, down from granular's
  24000: verl materialises full `(T x 152k)` logits per microbatch. This changes
  microbatch splitting only, not the gradient.

### Methodology note -- a correction worth remembering

Sweeps 1-4 were judged against `train10.log`/`sdoff_05.log`, which were
collected at `train_batch_size=64`. The sweep configs ran at 256. The apparent
"3x training slowdown" and "cliff at gpu_mem_util 0.6" were both artifacts of
that mismatch -- 4x the batch, not a regression. **Always confirm the batch size
of the reference run, not just of the run under test.** Scaling 64 -> 256 costs
2.57x, i.e. sublinear and healthy.

---

## Head-to-head vs granular-cais-rl (2026-08-16)

Both systems strictly sequential (granular run with `streaming=false`,
`dynamic_groups=false`, `min_train_world_size=4`, since FastRL's sync mode cannot
overlap rollout with training). Qwen2.5-Coder-7B-Instruct, batch 256 x 5,
4x L40S. granular run id `nooverlap_bsz256` (4 steps); FastRL `g1_mem07_sdoff`
(2 steps).

| metric | granular (no overlap) | FastRL (best) | ratio |
|---|---|---|---|
| step wall | **553.5s** (499-590) | **1436.8s** (1388/1486) | 2.6x slower |
| rollout / generation | 221.9s | 197.0s | 0.89x -- FastRL FASTER |
| training | 297.2s | 1213.5s | 4.1x slower |
| overlap | 0s | 0s | matched |

For reference, granular WITH streaming is 478s/step (`sql_optimized_35step`), so
disabling overlap costs it 1.16x -- less than its own timer model predicted.

**FastRL's generation is not the problem.** The entire gap is the training phase:

* ~1/4 of it is `old_log_prob` (276.7s), which verl recomputes every step and
  granular never pays: with `ppo_epochs=1` and `mini_batch == batch` the
  importance ratio is identically 1, so it is mathematically redundant. verl uses
  `rollout_log_probs` only for debug metrics.
* the other ~3/4 is `update_actor` itself: 936.8s vs granular's 297.2s **total**
  train latency (3.15x). granular packs samples into concatenated `(1,T)` rows
  with no pad tokens (flash-attn varlen) and runs cpu_adam; FastRL here is forced
  to `ppo_max_token_len_per_gpu=4096` + Ulysses SP=2 + param/optimizer offload,
  all of which the sweep showed are required, not chosen, on 44GB cards.

Caveat: 4 steps vs 2, and both systems vary per step (granular 499-590s; FastRL's
`update_actor` spans 912-1227s across identical configs). Treat 2.6x as
approximate. Both generation figures include multi-turn SQL env time.
