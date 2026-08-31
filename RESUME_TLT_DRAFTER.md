# RESUME: TLT (FastRL) opportunistic drafter training

**This file is the cross-node handoff for the FastRL/TLT baseline work.** It is
committed deliberately: the agent's memory store lives at
`~/.claude/projects/<abs-project-path-with-slashes-as-dashes>/memory/`, so it is
keyed to both the machine and the checkout location and does NOT follow a repo
to a new node. Anything that must survive a migration belongs in git, here.

Scope note: this lives in the FastRL checkout on purpose. The companion system
(granular-cais-rl) is a separate repo and deliberately carries none of these
TLT details.

Cross-references in `[[double brackets]]` are memory-store notes that may not
exist on this machine; the essentials are inlined.

**Stopped 2026-08-19** to free GPUs for a teammate; work is migrating to another
node. Read with [[project_engine_ab_sglang_vs_vllm]] and
[[project_fastrl_fair_baseline]].

**UPDATED 2026-08-25** — moving to Slurm (Oscar) because the pistachio GPUs are
contended. Read `## STATUS 2026-08-25` next; it supersedes several claims below,
including the section still titled "THE BLOCKER".

**UPDATED 2026-08-25 (later, on Oscar)** — ran the flagship config there. Read
`### UPDATE 2026-08-25 (Oscar)` inside the STATUS section first: Adaptive
Rollout Engine alone gets real acceptance (2.05-8.50) with the matched pair;
Adaptive Drafter training now hangs the whole node instead of just
underperforming.

**UPDATED 2026-08-26** — got a full clean 10-step production-batch (256x5) run
of the matched pair with Adaptive Rollout Engine (drafter training still off).
Read `### UPDATE 2026-08-26` below: three real bugs found and fixed (an EAGLE
buffer sized off the model's unused 128K native context instead of our 8K
working ceiling; a related SD admission-check margin; a wrong Hydra config
path for the NCCL collective timeout), and a real quality finding -- the
paper's own base-model pairing basically doesn't learn on this workload in 10
steps, unlike the Coder-Instruct target used elsewhere in this comparison.

---

## STATUS 2026-08-25 — read this before the older sections

### Done since the last update

**1. The SD-off 35-step e2e run COMPLETED.** This is the reference point for
everything else. Target `Qwen2.5-Coder-7B-Instruct`, `SPECULATIVE=false`,
4xL40S, 256x5.

* reward `critic/score/mean` **-0.168 (step 1) -> 0.562 (step 35)**, crossing
  positive at step 4, plateauing ~0.5-0.6 from step ~14. It learns.
* step time fell 1150s -> ~460s as responses shortened (mean response length
  755 -> 471 tok) and zero-signal filtering bit harder (step 35 dropped 188
  groups / 940 samples).
* FSDP checkpoint `output/fastrl_sql/FastRL-SQL/e2e35/global_step_35/`, merged
  to HF at `output/e2e35_merged_hf/` via
  `python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor
  --target_dir <out>`. Both live on pistachio only, NOT in git (15 GB / 86 GB).
* the one `Traceback` in that log is a torch DataLoader `__del__` teardown
  warning after training finished. Cosmetic.
* the config that produced it is in `examples/sql/TRAIN_CONFIG_TUNING.md`
  (commit `b0f583e`): SP2 + `forward_prefetch=True` + `TOKEN_BUDGET=4096` +
  `GPU_MEM_UTIL=0.6`. SP2 beat SP1 by 17% and SP4 by 26% — the earlier
  "ulysses SP2 is overly slow" hunch was wrong.
* `'Final validation metrics: None'` at the end — the end-of-run eval produced
  nothing. Unexplained, low priority, but do not quote a val number from it.

**2. THE OLD BLOCKER IS FIXED.** The section below titled "THE BLOCKER:
FastRL's SD crashes on multi-turn" is **stale**. Commit `0f7c378` fixed it
(`prepare_for_decode`'s `skip_prepare` keyed off the static engine-level
`spec_algorithm` rather than the per-batch adaptive decision, so batches the
adaptive gate downgraded arrived unprepared). Confirmed 2026-08-25: ~30 min of
multi-turn SQL rollout with speculation actively engaging, `enforce_eager=true`,
**zero tracebacks**. The `BS_THRESHOLD=1` side-step is no longer necessary.

**3. Drafter training now observed running end-to-end in multi-turn.** The
older note says only fix #2 was confirmed. A 2026-08-25 run logged the whole
handshake *and* the training side:

```
Worker 3 (DP=3) released all TP ranks
Starting training: DP ranks [3], process ranks [3] (1 total)
[EagleTrainer rank 3] activate_training_model enter training_ranks=[3]
Training worker 3 completed, broadcasting STOP_TRAINING ...
All workers completed, broadcasting BATCH_COMPLETE
```

That exercises #1 (the buffer had data to train on), #2 (handshake), #3 (the
trainer no longer refuses) and #4 (the step counter advanced far enough to fire
at interval 10). **Do not overclaim:** this proves the machinery *runs*. It does
NOT prove the gradient improved the drafter — acceptance never moved (below).

### The mismatch that was wrong all along

`examples/grpo_sql_baseline.sh` defaulted `MODEL_PATH` to
`Qwen/Qwen2.5-Coder-7B-Instruct` while `SPEC_MODEL_PATH` defaulted to
`mit-han-lab/Qwen2.5-7B-Eagle-RL` — **an EAGLE drafter whose frozen embedding
and LM head are tied to base `Qwen/Qwen2.5-7B`, a different model.** Every
earlier SD measurement on the SQL workload used this invalid pair, which is the
likely explanation for the old "SD is 1.8x SLOWER" result
([[project_fastrl_sql_gpu_validation]]). The paper's own pair is in
`examples/grpo_7B.sh`: target `Qwen/Qwen2.5-7B`, drafter
`mit-han-lab/Qwen2.5-7B-Eagle-RL`. Both are cached on pistachio under
`/local_nvme0/mborjigi/hf`.

### UPDATE 2026-08-25 (Oscar): Adaptive Rollout Engine alone DOES get real acceptance -- Adaptive Drafter training is what's actually broken

Two runs on Oscar (`scripts/run_tlt_flagship_slurm.sh` and its new sibling
`run_tlt_flagship_norafter_slurm.sh`), same matched pair, same smoke scale
(16x4, 2 steps), one knob different:

1. **Both features on** (job 5197256, `run_tlt_flagship_slurm.sh`): hung
   completely ~8min into step 1, right after the Adaptive Drafter's background
   training handshake finished (`Worker 0 cleaning up training`), immediately
   followed by a print never seen in any prior run here: `torch_memory_saver:
   Cannot pause allocation that is not active. tag=kv_cache`. GPU util on all 4
   GPUs dropped to 0% and stayed there for 12h+ with zero further log output;
   the node's process/thread table for the job's cgroup was fully exhausted
   (every `srun --overlap` exec failed to fork -- a real leak, not a plain
   deadlock). This is WORSE than the pistachio result recorded just below
   (which at least logged repeated `accept len: 1.00` readings before its
   1800s external timeout fired) -- that run may itself have been silently
   heading toward the same hang and simply gotten killed first. Root cause not
   confirmed: live diagnosis was blocked because the fork table was already
   exhausted by the time this was investigated. Best guess, unconfirmed: a
   state-tracking race between the Adaptive Drafter's background-training
   reentry and the per-round KV-cache pause/resume bookkeeping in
   `fsdp_sglang.py`'s `sleep()`/`wake_up()` -- the same general class of bug as
   the already-fixed `0f7c378` crash (adaptive per-batch/subsystem state vs.
   static engine-level state), just in the memory-saver path instead of
   `prepare_for_decode`.

2. **Adaptive Rollout Engine only** (job 5242734, `DRAFTER_TRAINING=false
   DRAFTER_COLLECT_SGL=false`, everything else identical): completed cleanly,
   rc=0, 8m18s, both steps. **Accept len ranged 2.05-8.50 across decode
   batches** -- not the flat 1.00 below. This confirms the leading hypothesis
   in the (now-partly-stale) section right below: **the old "SD 1.8x slower" /
   accept-len-1.00 result was an artifact of the wrong drafter pairing, not a
   fundamental limitation of the paper's Adaptive Rollout Engine or of pairing
   SD with a base `Qwen2.5-7B` target.** Reward moved -0.167 -> -0.071 over the
   2 steps (expected for a cold base model at 2 steps, nothing to read into
   it). Gen time was 91s (step 1) / 49s (step 2) for 16x4 -- no sign of the
   15-47 tok/s tail collapse seen in the run below.

**So: Adaptive Rollout Engine (matched pair) works. Adaptive Drafter training
is the actual open blocker**, and now looks like a real bug (resource leak on
Oscar) rather than only a "contributes nothing" performance dead end. The
non-RL control suggested below (`scripts/bench_speculative_decoding.py`, no
training loop) is still the right next step for isolating whether even THAT
subsystem's underlying weight-sync path is sound, but do it with
`DRAFTER_TRAINING` specifically in scope now, not general SD.

---

### UPDATE 2026-08-26: full 10-step production run (256x5), matched pair, Adaptive Drafter still off -- three bugs found and fixed, one real quality finding

Goal: measure reward, generation length, and step latency over 10 steps at
production batch size, matched pair (`Qwen/Qwen2.5-7B` +
`mit-han-lab/Qwen2.5-7B-Eagle-RL`), Adaptive Rollout Engine on, Adaptive
Drafter training off (per the 2026-08-25 finding above). Took 9 attempts to
get a clean run; three were real bugs worth fixing in-tree, the others were
config-path/syntax mistakes made along the way. Script:
`scripts/run_tlt_norafter_prod_slurm.sh` -- its header comment has the full
attempt-by-attempt log; this section is the summary.

**Bug 1 -- EAGLE draft's kv_indices buffer sized off the model's unused 128K
native context instead of our 8K working ceiling (real bug, fixed).**
`AsyncEngine(...)` in `sglang_rollout.py`'s `_init_inference_engine` never
passed `context_length`, so sglang's `ModelConfig` fell back to the model's
own native `max_position_embeddings` (131072 for `Qwen2.5-7B`, confirmed via
both the target's and drafter's `config.json`) instead of our actual
`max_model_len=8192`. That uncapped value flows straight into
`TritonAttnBackend.max_context_len`, which directly sizes the EAGLE draft's
per-decode-batch `kv_indices` buffer
(`triton_backend.py::init_forward_metadata`) as `speculative_num_steps *
batch_size * topk * max_context_len` -- ~16x (131072/8192) larger than
needed, matching the ~3.3 GiB CUDA OOM allocation observed at ~100-concurrent-
request batch sizes exactly. This is why the OOM hit at wildly different step
counts across attempts (0, 6, 7, 8, 9) with near-identical memory numbers
each time -- not a leak, a single oversized transient allocation whose
success depended on the fragmented pool's momentary layout. Fixed by passing
`context_length=self.config.max_model_len` (+ bug 2's margin below) to
`AsyncEngine(...)`. Verified every other `context_len` consumer in sglang
(10 call sites checked) is unaffected for this workload -- see the fix's
comment in `sglang_rollout.py` for the full per-site reasoning.

**Bug 2 -- SD verify-token admission margin verl doesn't account for (real
bug, fixed).** Fixing bug 1 alone immediately exposed a second, previously-
invisible issue: sglang's own request-admission check
(`tokenizer_manager.py::_validate_one_request`) adds a `reserve_input_token_num`
margin (= `speculative_num_draft_tokens`, 48 in our config) on top of
input+max_new_tokens before comparing to `context_len`, to cover the
draft/verify process occasionally emitting a few tokens past the strict
`max_new_tokens` boundary. verl's own `max_new_tokens` computation
(`sglang_rollout.py`'s `max_new_tokens = min(response_length, max_model_len -
len(generation_prompt_ids) - 1)`) never knew about that margin -- invisible
when `context_len` was accidentally 131072 (huge slack), immediately hit once
`context_len` was correctly capped to 8192 ("Requested token count exceeds
the model's maximum context length of 8192 tokens ... 5582 input + 2657
completion"). Fixed by widening `context_length` by that same reserve when SD
is on (8192 -> 8240, ~0.6%, negligible next to the buffer bug 1 was already
shrinking 16x).

**Bug 3 -- wrong Hydra config path for the NCCL collective timeout (mistake,
not a code bug, fixed in the script).** The DP-straggler `dist.barrier()`/
`ALLREDUCE` timeout from the 2026-08-25 flagship-smoke work (attempt 4 in the
script header) recurred at production scale. `trainer.nccl_timeout` looked
like the right Hydra key (it's a documented field in
`fastrl_trainer.yaml:184`), but the code that actually reads it
(`fsdp_workers.py`'s `self.config.get("nccl_timeout", 600)` inside
`ActorRolloutRefWorker.__init__`) reads from `self.config` =
`config.actor_rollout_ref` as a whole (see `ray_trainer.py:869,889`,
`config=self.config.actor_rollout_ref`) -- a sibling key to `.actor`/
`.rollout`/`.model`, NOT `config.trainer`. `trainer.nccl_timeout=1800` parsed
fine and silently did nothing (the log kept showing `Timeout(ms)=600000`).
Correct override: `actor_rollout_ref.nccl_timeout=1800` (no `+` prefix --
Hydra confirmed the key already exists once given the right path).

**Result (job 5359188, rc=0, 3h38m):** all 10 steps completed, mean step time
~1399s (range 1157-1487s), no OOM, no rejection, no collective timeout.
**Reward stayed flat at -0.75 to -0.80 across all 10 steps** -- essentially no
learning, vs. the SD-off Coder-Instruct reference's -0.168 -> 0.562 climb over
35 steps. **252 of 256 GRPO groups were dropped as zero-signal on every single
step** (nearly all groups score identically across their 5 rollouts -- almost
no training signal reaches the optimizer). **Response length stayed
~2300-2460 tokens with 60-80% of responses hitting the 3000-token cap every
step** -- the base (non-instruct) model isn't reliably terminating with
`<solution>`, unlike the Coder-Instruct target. Consequently latency never
improves either: no downward trend across the 10 steps, unlike the reference
run's 1150s -> ~460s fall as responses shortened with learning -- this run
never converges enough to earn that speedup. This confirms and quantifies the
"tension nobody has resolved yet" section below: paper fidelity (base
`Qwen2.5-7B` + their drafter) buys a working, stable SD setup, but at a real
cost in this workload's actual training dynamics, at least within the first
10 steps.

Per-step metrics only landed for 6 of 10 steps (1, 3, 4, 6, 7, 10) -- the same
intermittent console-logging gap noted in earlier attempts, confirmed here to
happen even on completely healthy, non-crashing steps (occurs for both
2026-08-25 and 2026-08-26 runs alike). Not investigated further; per-step wall
time is still fully reconstructible from tqdm's own timestamps regardless.

---

### UPDATE 2026-08-29: the "old_log_prob/update_actor 15-20x slower without SD" number wasn't a separate bug -- it's the SAME zero-signal-collapse finding; plus a real single-turn hang bug found and fixed; plus the multi-turn hypothesis is NOT yet confirmed

**Clean SD on/off ablation** (same base `Qwen2.5-7B`, same 256x5 batch, only
`speculative.enable` flipped, `scripts/run_tlt_sdoff_ablation_slurm.sh` vs.
`run_tlt_norafter_prod_slurm.sh`): SD OFF learns (reward -0.32 -> +0.45 over 10
steps, zero-signal groups dropping from 52/256 toward low double digits) while
SD ON stays flat (-0.80 to -0.75, 252/256 dropped on every single step) --
this is the real, confirmed effect of SD on this workload's training dynamics,
not just a latency question.

**The apparent "old_log_prob/update_actor 15-20x slower without SD" anomaly
noted when this ablation was first run is NOT a separate bug.**
`filter_zero_signal_groups` runs in `ray_trainer.py` *before*
`old_log_prob`/`update_actor` (see `# Drop zero-signal groups before any
training-side compute` at that call site), so those two phases only ever see
the *surviving* samples -- next to nothing when SD collapses 252/256 groups,
proportionally far more when only 52/256 collapse. `perf/total_num_tokens`
(and the wall-clock generation numbers) are measured on the *pre-filter*
batch (`batch.meta_info["global_token_num"]` is set earlier, at the top of
the reward block), so it reflects rollout volume, not training volume --
comparing SD-on's "few total tokens, fast training" against SD-off's "more
total tokens, slow training" was comparing the wrong pair of numbers. Same
underlying phenomenon as the diversity-collapse finding, showing up twice.

**A real, separate bug found and fixed: single-turn generation
(`multi_turn.enable=False`) hung indefinitely on every run, with or without
SD.** `_generate_with_drafter` (`sglang_rollout.py`, the single-turn
generation path) called `self.sharding_manager.release_memory()`
unconditionally after every generation, with no gate on whether opportunistic
drafter training was actually active -- unlike its multi-turn counterpart
`_drafter_release_after_generation`, which correctly gates the identical call
behind `_drafter_training_active()` (`if not
self._drafter_training_active(is_validate): return`). With
`DRAFTER_TRAINING=false` (every run in this file), the multi-turn path's gate
makes that call a no-op, so the KV cache/weights get released exactly once,
by the training loop's own round-transition `rollout.sleep()`. The
single-turn path had no such gate, so it released early on every call --
then the training loop's own `sleep()` tried to release the *same* tag again,
hit `torch_memory_saver: Cannot pause allocation that is not active`, and the
run hung for hours with zero further progress (reproduced identically with
`SPECULATIVE=true` and `SPECULATIVE=false`, confirming it's unrelated to SD).
Fixed by adding the same `_drafter_training_active()` gate to
`_generate_with_drafter`'s early-release block (threading `is_validate`
through from its one call site). Verified: both single-turn smoke jobs now
complete cleanly (rc=0) where they previously hung for their full 3h
timeouts.

**With the hang fixed, tested whether the zero-signal collapse is specific to
multi-turn (hypothesis: a bug in the SQL/multi-turn integration, not a
general SD effect) -- NOT CONFIRMED at the scale tested.** Same task/model/
drafter, `multi_turn.enable=False`, 16x4 smoke scale (`scripts/
run_tlt_singleturn_sdon_slurm.sh` / `_sdoff_slurm.sh`):

* SD ON: 14/16 groups dropped (87.5%)
* SD OFF: 13/16 -> 14/16 groups dropped (81-87.5%) across 2 steps

Essentially the same collapse rate with or without SD -- nothing like the
stark 252/256-vs-52/256 gap seen in multi-turn at production scale. Response
length is also much shorter here (mean ~290-300 tokens vs multi-turn's
~2300-2450), consistent with the untrained base model just performing
uniformly poorly at single-shot SQL regardless of SD, rather than SD
specifically destroying diversity. Two live possibilities, not yet
distinguished:

1. The multi-turn SD-vs-diversity gap is real but doesn't show up at 16
   groups (need a production-scale, i.e. 256x5, single-turn run for a fair
   comparison against the multi-turn numbers -- not yet attempted: the first
   production-scale single-turn attempt, before the hang fix, also hit a
   separate long-tail problem at that batch size and was abandoned in favor
   of the smoke-scale test above).
2. The diversity-collapse gap is itself specific to the multi-turn
   integration (tool-call/observation injection interacting with speculative
   decoding's draft/verify cycle in some way single-turn never exercises),
   and 16 groups was simply too small a sample either way.

Next step to resolve this: rerun the single-turn SD on/off pair at production
batch size (256x5) now that the hang is fixed, watching for the same
long-tail slowness the first attempt hit (single-turn has no natural
per-turn checkpoint the way multi-turn does, so a handful of straggler
trajectories running to the full 3000-token cap can dominate wall time --
worth a shorter `max_response_length` for this specific probe if it recurs).

---

### UPDATE 2026-08-30 (pistachio): question answered -- the collapse is NOT multi-turn-specific, and it is NOT a GRPO diversity collapse. SD stops generations ever closing `</sql>`.

Ran the single-turn 256x5 SD on/off pair this file asked for, then a 2x2
rollout-only probe (turn mode x SD) that measures the trajectory *text*
rather than the zero-signal counter. Both arms of the training ablation come
from ONE parameterised script (`scripts/run_tlt_singleturn_prod_local.sh`,
`SPECULATIVE` is the only input that differs) rather than two hand-synced
scripts -- the previous single-turn pair had already drifted, only the SD-on
one carrying the `nccl_timeout` override.

**1. The collapse reproduces in single-turn, so it is not the multi-turn
integration.** Matched 5 steps each, `Qwen/Qwen2.5-7B` +
`mit-han-lab/Qwen2.5-7B-Eagle-RL`, everything else identical:

| single-turn 256x5 | zero-signal groups per step | steps with NO update at all |
|---|---|---|
| SD OFF | 236, 236, 224, 216, 188 / 256 | 0/5 |
| SD ON  | 256, 256, 256, 256, 256 / 256 | **5/5** |

SD-off score -0.699 -> -0.705 with the drop count falling 92% -> 73% (slow
learning starting). SD-on never completed a single optimizer step in 5 steps.
Step time 309s (off) vs 755s (on) -- **2.44x slower**.

Note on acceptance, because it contradicts the older "blocker" section below:
**speculation is working normally in these runs.** Over all 383 decode batches
of the SD-on arm, accept len is mean **2.154**, median 1.61, p90 3.80, max
5.52, and only 70/383 batches sit at exactly 1.00 (the multi-turn probe is
mean 2.203, max 9.00). That matches the 2026-08-25 Oscar result (2.05-8.50)
and confirms the `accept len 1.00` reading from the old pistachio flagship run
was the wrong-drafter artifact, not a property of this pair. It also makes the
2.44x slowdown its own oddity -- SD with a mean acceptance above 2 should be
*faster*, not 2.4x slower -- and it means everything below is happening with a
healthy, functioning drafter, not a degenerate one.

**2. "Per-step metrics sometimes don't print" is NOT cosmetic -- it is the
signature of a totally-collapsed step.** The 2026-08-26 section above records
metrics landing for only 6 of 10 steps and calls the gap harmless, "confirmed
to happen even on completely healthy, non-crashing steps". It isn't:
`ray_trainer.py`'s `drop_zero_signal` branch, when `n_grp == -1` (every group
zero-signal), prints `[drop_zero_signal] step N: every group was zero-signal`,
bumps the step and `continue`s -- never reaching the metrics assembly or the
console logger. A missing metrics line therefore *means* 256/256 collapsed
with no update. So the multi-turn SD-on production run was worse than
recorded: 252/256 on the six steps that logged, and 256/256 on the four that
did not. Grep `every group was zero-signal`, not just the metrics line.

**3. Root cause: SD stops generations from ever closing `</sql>`.** 2x2
rollout-only probe (`scripts/run_tlt_diversity_probe.sh` +
`scripts/analyze_group_diversity.py`), 1280 trajectories per cell, same base
checkpoint and prompts, dumped pre-filter:

| cell | resp_chars | turns/traj | `</sql>` missing | `<solution>` present | mean uniq /5 |
|---|---|---|---|---|---|
| multi-turn, SD off | 3716 | **3.08** | 6.0% | **54.0%** | 4.99 |
| multi-turn, SD on  | 12373 | **0.97** | **94.0%** | 2.1% | 5.00 |
| single-turn, SD off | 1270 | 0.00 | 7.3% | 2.0% | 4.98 |
| single-turn, SD on  | 4473 | 0.01 | **91.4%** | 3.4% | 5.00 |

Turning SD on makes responses ~3.3x longer and takes `</sql>` closure from
6-7% missing to 91-94% missing, in BOTH turn modes. In multi-turn that severs
the agent loop: turns/trajectory 3.08 -> 0.97, half of all trajectories never
complete even one tool round-trip, `<solution>` production falls 54% -> 2%,
so essentially every rollout takes `sql_reward.py`'s -1.0 format-error path,
every group in the batch scores identically, and every group is dropped.

**This means the "SD destroys GRPO sampling diversity" framing earlier in this
file is wrong.** Within-group diversity is *perfect* in all four cells --
`exact_dup_rate` 0.000 everywhere, mean unique completions 4.98-5.00 out of 5.
The rollouts are all different from each other; they are uniformly *unscorable*.
It is a termination/format collapse that yields uniform -1.0 rewards, not a
diversity collapse -- which matters because the fixes are completely
different, and because the zero-signal counter alone cannot tell the two
apart (that is what `analyze_group_diversity.py` exists to separate).

**Why this is a correctness bug, not a tuning problem.** Correct speculative
decoding is distribution-preserving *at any acceptance rate*: the whole point
of the draft/verify rejection-sampling scheme is that the emitted sequence is
drawn from the target model's distribution, so accepting more drafted tokens
buys latency and nothing else. Enabling it must not change what the model
writes. Here it changes the output enormously -- 6-7% -> 91-94% of
trajectories never closing `</sql>`, responses 3.3x longer -- with everything
else held fixed (same base checkpoint, same prompts, same sampling params,
same caps, single rollout step, no training in between). And it does so with
a healthy drafter (mean accept len 2.15), so this is not a
degenerate-speculation artifact. Something in the verify/accept path is
emitting tokens the target model would not have emitted.
Ruled out while chasing this, all worth not re-checking:
* Relaxed acceptance thresholds -- `speculative_accept_threshold_single/acc`
  both default to 1.0 (strict) and are not set anywhere here.
* The greedy-verification fallback -- `eagle_info.py` falls back to
  `verify_tree_greedy` when `not TREE_SPEC_KERNEL_AVAILABLE`, which would
  collapse a GRPO group outright; but that flag is `is_cuda()`, true here.
* Sampling-params plumbing in the verify path -- temperature IS applied
  (`next_token_logits / expanded_temperature`) and top-k/top-p are both
  renormalised before rejection sampling.
* Missed stop-string detection under multi-token steps -- `check_finished`
  runs after all accepted tokens are appended and tests `stop_str in
  self.decoded_text`, and anyway the tag is *absent* from the text, not
  overshot, so nothing was missed: it was never generated.

That leaves the tree acceptance/residual sampling itself
(`tree_speculative_sampling_target_only`, called with `draft_probs` all
zeros) as the prime suspect. Next step is a non-RL control: generate a fixed
prompt set through plain sglang with this pair, SD on vs off, and compare
`</sql>` closure and length -- if it reproduces with no training loop at all,
it is purely an engine bug and can be reported/fixed as one.

**4. New bug found: a 2-token KV-pool leak kills the scheduler on the
single-turn path.** `scheduler.py::check_memory`, run from
`self_check_during_idle()` between steps, is a strict equality test and fired
at step 6 of 10 with `max_total_num_tokens=135945` vs
`available_size=2879 + evictable_size=133064` = 135943 -- two tokens, 0.0015%,
which cannot exhaust anything, but it raises and takes the whole job down
(scheduler dead, GPUs idle, run hung). Multi-turn production runs never hit
it, so it looks specific to the single-turn path -- the same under-tested path
that produced the `release_memory()` hang fixed in `b793853`. NOT fixed: made
tolerable behind opt-in `SGLANG_TOLERATE_KV_LEAK=1` (off by default, set by
both ablation scripts so the arms stay symmetric) purely so a diagnostic can
finish. The accounting bug still wants a root cause.

**Also worth knowing:** single-turn is a poor discriminator in absolute terms
regardless of SD -- with SD off it still drops 92% of groups, because the SQL
prompt is agentic and only 2.0% of one-shot rollouts ever emit `<solution>`
(vs 54.0% in multi-turn). That is a prompt/protocol artifact, not evidence
about SD, and it is why the 16-group smoke looked inconclusive. The SD effect
is still visible on top of it (92% -> 100%, and 0/5 -> 5/5 dead steps).

Reproduce:

```bash
SPECULATIVE=false MAX_STEPS=10 bash scripts/run_tlt_singleturn_prod_local.sh
SPECULATIVE=true  MAX_STEPS=10 bash scripts/run_tlt_singleturn_prod_local.sh
for mt in true false; do for sd in true false; do
  MULTI_TURN=$mt SPECULATIVE=$sd bash scripts/run_tlt_diversity_probe.sh
done; done
```

The multi-turn SD-on probe OOMs at `GPU_MEM_UTIL=0.5` (Triton CUDA OOM, it is
the longest-generation cell); it completes at `GPU_MEM_UTIL=0.4`.

---

### UPDATE 2026-08-30b: root-cause hunt. SD is injecting junk tokens, it is NOT a sampling-distribution effect, and eight candidate causes are eliminated.

Follow-up to the above, chasing "SD should only change decode latency". It
does not: **SD emits tokens the target model never selected.** Reading the
actual dumped text rather than the aggregate metrics makes it obvious:

```
SD-off : <think>I will first check if the 'Apartments' table exists and contains entri
SD-on  : <think>ük>First teh, I will checkük if the 'ükApartükmentsük' table exists Qty
```

A foreign token is spliced repeatedly into otherwise-coherent text, which then
degenerates into repetition loops. Counting non-ASCII fragments over the 256
greedy prompts: **SD-on 23,048, SD-off 2**, dominated by `哈哈哈哈` and `啊`.

**The decisive experiment is greedy decoding.** Under sampling, two runs
legitimately differ token-for-token (different RNG streams), so no
distributional argument is clean. Under temperature 0 correct speculative
decoding is *bit-identical* to non-speculative decoding -- a draft token is
accepted iff it equals the target's argmax -- so any difference is a definite
bug. Greedy, single-turn, 256 prompts:

| greedy | `</sql>` missing | resp_chars | unique of 5 |
|---|---|---|---|
| SD OFF | **3.9%** | 1041 | 1.50 |
| SD ON  | **91.1%** | 6599 | **4.66** |

Two things follow. The corruption survives greedy, and greedy runs a
*different* kernel (`verify_tree_greedy`, not
`tree_speculative_sampling_target_only`), so the stochastic accept math is not
the cause. And `unique of 5 = 4.66 at temperature 0` means SD-on returns a
*different* answer almost every time for the same prompt, where SD-off is
near-deterministic at 1.50 -- so the defect is **nondeterministic and
batch/state-dependent**, not a mis-specified distribution. That is the
signature of cross-request state corruption, which also fits the 2-token KV
allocator leak reported above firing on this same path.

**Eliminated, with the evidence, so nobody re-chases them:**

1. *Stochastic accept kernel / rejection-sampling math* -- corruption is
   identical under greedy, which does not use that kernel.
2. *`enforce_eager` / CUDA graphs off* (hypothesis 3 of the old blocker
   section) -- re-ran with `ENFORCE_EAGER=false`: `</sql>` missing 92.0% vs
   91.4% eager. No effect. CUDA-graphs-off is NOT in conflict with TLT.
3. *Uninitialised `predict` buffer* -- `eagle_info.py` allocates it with
   `torch.empty`, so an `accept_index` pointing at an unwritten slot would
   read junk. Tested by filling it with a known sentinel token
   (`FASTRL_PREDICT_SENTINEL=9008`, `' Initialize'`): the sentinel does NOT
   appear in the output (1 occurrence) and the junk persists (8,748
   fragments). The kernel does write those slots -- with wrong ids. Refuted.
4. *The `context_length` cap from `75df8ae` shrinking the EAGLE `kv_indices`
   buffer* -- the buffer is `bs * topk * max_context_len` wide and the slice
   needs `seq_lens_sum * topk + bs*(i+1)`, and every `seq_len <= context_len`
   by admission, so it cannot overflow. (The A/B via
   `FASTRL_ENGINE_CONTEXT_LEN=131072` OOMs, which is exactly the bug that
   commit fixed, so it is not runnable at this batch size anyway.)
5. *Relaxed acceptance thresholds* -- `speculative_accept_threshold_single`
   and `_acc` both default to 1.0 (strict) and are set nowhere.
6. *The greedy-verification fallback* -- would collapse a group outright, but
   it is gated on `not TREE_SPEC_KERNEL_AVAILABLE` = `not is_cuda()`, false here.
7. *Sampling-params plumbing in the verify path* -- temperature IS applied
   (`next_token_logits / expanded_temperature`), top-k and top-p are both
   renormalised, `top_k=-1` normalises to `TOP_K_ALL` (a no-op), and the run
   sets no penalties or `min_p`. Matches the normal sampler's effective behaviour.
8. *Drafter holding stale embed/lm_head* -- `eagle_worker.py` shares the
   target's embed and head **by reference at init**
   (`get_embed_and_head()` -> `set_embed_and_head`), and verl inits the target
   with `load_format='dummy_dtensor'` (random) and syncs real weights after,
   so this looked like the answer. But sglang's `default_weight_loader` does
   `param.data.copy_()` -- in-place -- so the shared references do see the
   real weights. Refuted (worth re-checking if a non-default loader path is
   ever used).

Also worth recording: `accept len` is **not** 1.00 in these runs (mean 2.15
sampled, 1.49 greedy, max 5.52), so this is a *functioning* drafter corrupting
output, not a degenerate one.

**Where to go next**, in order:

1. **Standalone sglang control, no verl.** Same target+drafter, real
   checkpoint (no `dummy_dtensor`), greedy, one fixed prompt set, SD on vs
   off. This is the one test that separates "sglang/EAGLE bug" from "verl
   integration bug" and it has been on the list since 2026-08-25 without being
   run. Everything above is inside the RL harness.
2. **Concurrency sweep.** Corruption is nondeterministic and batch-dependent,
   so drive `max_running_requests` down to 1 and back up. If it disappears at
   1, it is definitively cross-request state contamination in the tree/KV
   plumbing, which would also explain the 2-token allocator leak, and the two
   should be fixed together.
3. Only then dig into the tree/KV bookkeeping itself (`eagle_info.py`'s
   `verify` -> `accept_index` / `evict_mask` / `assign_req_to_token_pool`).

Repro for the greedy pair and the diagnostics:

```bash
MULTI_TURN=false SPECULATIVE=false GREEDY=1 bash scripts/run_tlt_diversity_probe.sh
MULTI_TURN=false SPECULATIVE=true  GREEDY=1 bash scripts/run_tlt_diversity_probe.sh
.venv/bin/python scripts/compare_runs_bitwise.py \
  output/tlt-probe-st-sdoff-greedy/rollouts/step1.jsonl \
  output/tlt-probe-st-sdon-greedy/rollouts/step1.jsonl
```

`compare_runs_bitwise.py` prints exact-match rate and the first divergence per
prompt; the archived dumps for every cell are in `output/probe_archive/`
(not in git). Both diagnostic env vars (`FASTRL_PREDICT_SENTINEL`,
`FASTRL_ENGINE_CONTEXT_LEN`) are unset by default and change nothing when unset.

---

### THE CURRENT BLOCKER (pistachio, 2026-08-25, superseded by the Oscar runs above for the "does SD itself work" question): matched pair, and speculation still contributes nothing

Ran the full flagship config (paper's matched pair, Adaptive Drafter + Adaptive
Rollout Engine both on, `ENFORCE_EAGER=true`, 16x4, `GPU_MEM_UTIL=0.5`):

* **`accept len: 1.00, accept rate: 0.11` on every single decode batch.**
  accept len 1.00 means the target's own bonus token and nothing else — zero
  drafted tokens ever accepted, i.e. SD is pure overhead.
* consequently slow: ~170 tok/s at bs=16 falling to 15-47 tok/s on the tail. The
  2-step 16x4 smoke **timed out at 1800s (rc=124) without finishing step 2**,
  where the SD-off run does full 256x5 steps in ~460s.
* no crash, no OOM, no traceback. A performance/correctness dead end, not an
  instability.

A flat, exact 1.00 on every batch is the signature of "no draft token is ever
accepted", which even a badly distribution-mismatched drafter should beat. So
**suspect plumbing before suspecting drafter quality.** Ranked hypotheses, none
yet tested:

1. **RL weight sync desyncs the drafter.** The target's weights are pushed to
   the engine every step; if the drafter's frozen embed/LM-head references are
   not re-tied (or are re-tied to the *updated* target while the drafter body is
   stale) acceptance would collapse to zero. Highest prior, because it is
   specific to the RL setting — which is exactly what is unusual here versus
   every static-inference EAGLE deployment.
2. **Sampling params.** Ours is `temperature=0.6, top_p=0.95` (matching
   granular); the paper used `temperature=0.9`. Cheap to test, unlikely to
   explain an exact 1.00.
3. **`enforce_eager` degeneracy in tree verification.** SD provably does not
   *crash* without CUDA graphs, but the verify path is far less exercised there.
   Test by flipping `ENFORCE_EAGER=false` purely as a diagnostic — if acceptance
   jumps, then "CUDA graphs off" and TLT's flagship feature are in direct
   conflict, which is itself a finding worth reporting.
4. **Distribution mismatch.** The drafter targets the paper's post-RL math/code
   model; ours is multi-turn SQL. Real, but should degrade acceptance, not zero it.

Diagnose with a **non-RL control first** — it separates hypothesis 1 from the
rest in one shot: run `scripts/bench_speculative_decoding.py` (or plain sglang)
with that target+drafter pair and no training loop. Healthy acceptance there and
1.00 inside RL means it is the weight sync.

### The tension nobody has resolved yet

Matching the paper means `Qwen/Qwen2.5-7B`, a **base** model. Our SQL pipeline
feeds a chat template and the SkyRL-SQL recipe normally starts from an
instruct/coder checkpoint — the completed 35-step run above used
`Qwen2.5-Coder-7B-Instruct` and learned well. Base-model rollouts on this
workload may simply be poor, confounding any speed comparison with a quality
difference. Decide explicitly which is held fixed:

* **paper fidelity** -> `Qwen2.5-7B` + their drafter, accepting that reward
  quality may be far worse than the 0.56 reference.
* **workload fidelity** -> `Qwen2.5-Coder-7B-Instruct`, and then a matching
  drafter must be TRAINED for it (`eagle-train/`, ~2h) because the published one
  does not apply. More honest for our paper, but costs a drafter-training cycle.

### How to resume on Slurm

`scripts/run_tlt_flagship_slurm.sh` is the flagship job, written for Oscar and
matching `run_35step_train_slurm.sh`'s conventions (nproc ulimit, Lmod CUDA,
`RAY_OVERRIDE_RESOURCES`). Smoke first, always:

```bash
sbatch --export=ALL,SMOKE=1 scripts/run_tlt_flagship_slurm.sh   # 2 steps, 16x4
sbatch scripts/run_tlt_flagship_slurm.sh                        # 35 steps, 256x5
```

The smoke variant exists precisely because the full-feature config timed out at
that scale on pistachio — **treat "completes 2 steps" as the gate** for the
20-hour job. Triage commands are in the script's footer; the first thing to look
at is `grep -o 'accept len: [0-9.]*'`. If it is still 1.00, do not bother reading
the timings: fix acceptance first, because a speed number for a drafter that
accepts nothing measures nothing about TLT.

`Qwen/Qwen2.5-7B` (15 GB) and `mit-han-lab/Qwen2.5-7B-Eagle-RL` (1.5 GB) are
cached on pistachio at `/local_nvme0/mborjigi/hf`; on Oscar they re-download to
`$HF_HOME` unless copied over.

---

## Where this stands (2026-08-19 — see STATUS above for what has changed since)

**DONE:** engine A/B — vLLM ~7% faster than sglang on a bit-identical workload,
so the 2.6x system gap is framework not engine. Fully written up; no further
work needed. See [[project_engine_ab_sglang_vs_vllm]].

**IN PROGRESS:** making TLT's headline feature (opportunistic drafter training)
actually run, so the baseline gets a fair shot at its main contribution in
multi-turn — which is the whole point of the comparison.

## The 4 fixes — fastrl commit `e945e8f`

Drafter training could not run as shipped. **Three of the four break single-turn
too**, so this path looks unexercised rather than merely multi-turn-hostile.

1. `add_drafter_data_to_buffer` read `self.ulysses_sharding_manager`, but
   `drafter_module` is assigned to `self.rollout_sharding_manager`;
   `FSDPUlyssesShardingManager` has no such attribute -> returned early EVERY
   call, collector silently added nothing. (both modes)
2. Coordinator handshake existed only in `_generate_with_drafter` (single-turn).
   Added `_drafter_release_after_generation` / `_drafter_complete_batch` to
   `_req_level_generate_sequences`.
3. `_training_step_impl` refused whenever `collect_hidden_states_from_sgl` was
   false, even with a full buffer -> now gates on data availability. (both modes)
4. `increment_rl_step` = registered RPC with zero callers -> `current_rl_step`
   stuck at 0 -> `should_collect_data_this_step()` false for any interval > 1.
   Now called from `fit()`. (both modes)

**Validation status — fix #2 CONFIRMED, others NOT:**
`output/drafter_val5/run.log` reached
`Starting training: DP ranks [1], process ranks [1]` and
`EagleTrainer rank 1] activate_training_model` **in multi-turn**, which is only
reachable via the new handshake. Killed before `old_log_prob`, so #1/#3/#4 are
still unverified (need a completed step: buffer adds + an executed training step
+ acceptance-length movement).

**Instrumentation gap to fix first:** `_drafter_release_after_generation` logs
only on exception, so success is silent (`[drafter]` count was 0 despite
working). Add an info log on the success path or resuming will be confusing.

## ~~THE BLOCKER~~: FastRL's SD crashes on multi-turn when speculation engages

> **STALE as of 2026-08-25 — FIXED by commit `0f7c378`.** Kept for the
> diagnosis trail only. Multi-turn SD now runs for 30+ min with
> `enforce_eager=true` and zero tracebacks; `BS_THRESHOLD=1` is no longer
> needed to side-step it. The live blocker is zero draft acceptance — see
> `## STATUS 2026-08-25`.

Reproduced 3x, **with drafter training OFF** (so independent of the 4 fixes):
* CUDA graphs on -> `size of tensor a (20) must match tensor b (2635)` — the
  exact assert `examples/grpo_sql_baseline.sh` already documents.
* `enforce_eager=True` -> `set_kv_buffer` gets cache locations sized as the
  WHOLE KV pool (`expanded size of the tensor (127619|167205) must match ...
  (4096)`), in `triton_backend.forward_decode`.
Not caused by `bs_threshold` (fails at 32 and 64 alike — I initially blamed 64,
wrong).

**Strong inference (NOT directly measured):** at 256x5 the adaptive gate kept
speculation essentially OFF, which is why those sweeps completed. Re-enabling
needs **10 consecutive** decode batches at/below `bs_threshold`
(`eagle_worker.py:196 adaptive_spec_warmup_checks = 10`); at ~1280 concurrent
the tail never accumulates that. At 8x5 everything is below threshold, SD
engages immediately, and it dies. If true, **restate the old conclusion**: not
"SD is a net loss at 256x5" but "SD barely ran at 256x5, and crashes where it
does run on multi-turn." Old logs lack decode stats — confirm by logging
accept_length or the "Speculative decoding will be disabled" message.

## How to resume (2026-08-19 version — superseded)

> **Superseded 2026-08-25.** Both goals of this recipe are now met: the SD
> multi-turn crash is fixed (`0f7c378`) and drafter training has been observed
> running end-to-end, so `BS_THRESHOLD=1` is no longer needed. Use
> `### How to resume on Slurm` in `## STATUS 2026-08-25` instead. Kept because
> the checks it lists are still the right ones for verifying the 4 fixes, and
> the pistachio-local drafter path here is still the only trained
> Coder-Instruct-matched drafter that exists.

Validate the plumbing while side-stepping the SD bug: **`BS_THRESHOLD=1`** keeps
`use_spec` true (drafter + optimizer built, handshake armed) while the gate keeps
speculation from engaging. That is the run that was working when stopped:

```bash
cd /local_nvme1/mborjigi/fastrl
setsid nohup env CUDA_VISIBLE_DEVICES=2,3 \
  NGPUS=2 TP=1 TRAIN_PROMPT_BSZ=8 MINI_BSZ=8 N_RESP=5 MAX_STEPS=4 \
  MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=1024 GPU_MEM_UTIL=0.45 ENFORCE_EAGER=True \
  SPECULATIVE=true SPEC_MODEL_PATH=/local_nvme1/mborjigi/eagle-drafters/sql-cold-e3 \
  BS_THRESHOLD=1 DRAFTER_TRAINING=True DRAFTER_INTERVAL=1 \
  ULYSSES=2 TOKEN_BUDGET=2560 \
  TRAIN_FILE=data/skyrl_sql/bench650.parquet VAL_FILE=data/skyrl_sql/validation.parquet \
  EXPERIMENT_NAME=drafter_val5 \
  bash examples/grpo_sql_baseline.sh > output/drafter_val5/run.log 2>&1 < /dev/null &
```
Check: `samples to drafter DataBuffer` (#1), `Starting training: DP ranks` (#2),
an EagleTrainer step completing (#3), `current_rl_step` advancing (#4).
Then: fix the SD multi-turn crash, then sweep `bs_threshold` for real.

## Costs measured (size the real run with these)

* Drafter training adds ~7 GiB/GPU: actor process 15.03 GiB with it vs ~7.9 GiB
  without, on top of sglang's 27.88 GiB. OOMs a 44 GiB L40S at
  `GPU_MEM_UTIL=0.5` + 16x5. Needed 0.45 + 8x5 + 1024-token responses to fit.
* Ulysses SP=2 will NOT init on 3 GPUs (device-mesh IndexError). Use
  `ULYSSES=1`, and then `TOKEN_BUDGET >= max_prompt + max_response`.

## Node migration inventory

**CORRECTION 2026-08-19: both repos point at the user's OWN forks**
(`git@github.com:mbwangfpdc/fastrl.git`,
`git@github.com:mbwangfpdc/granular-cais-rl.git`), so this migrates by push/pull
— an earlier note in this file claimed fastrl's origin was mit-han-lab and that
pushing was forbidden. That was stale.

* `/local_nvme1/mborjigi/granular-cais-rl` — already pushed, `git pull` on the
  new node gets everything.
* `/local_nvme1/mborjigi/fastrl` @ `0e73bc2` — commits `8757d9d`, `f96827d`,
  `e945e8f`, `0e73bc2` **were still unpushed when this was written** (the push
  was blocked by the permission classifier). Push before migrating, or those 4
  fixes + the trace sink exist on this node only.
* `/local_nvme1/mborjigi/granular-cais-rl` — my engine-A/B work is commits
  `48ccb7e`, `d9d713b`, `22e90bc`; HEAD has since moved on (gantt commits
  from another session), so bundle/copy the branch, not a single commit.

**NOT in git — copy or regenerate:**
* `/local_nvme1/mborjigi/eagle-drafters/` — trained drafters (2.9 accept
  length), ~2h to retrain, NOT reproducible otherwise. Use
  `eagle-train/scripts/slim_drafter.py` (fastrl `0e73bc2`): `strip` drops the
  frozen target-derived `embed_tokens` (1.61 GB -> **517 MB**), `rehydrate`
  re-attaches it from `Qwen/Qwen2.5-Coder-7B-Instruct` on the far side —
  verified bit-identical. `sql-cold-e3-slim/` is already stripped and ready to
  move. 517 MB still exceeds GitHub's 100 MB file cap, so move it by
  scp/rsync/HF-hub, not git. **Use ABSOLUTE paths** -- a relative `--slim`
  resolves against the CWD and fails confusingly (fixed in fastrl `6595afe` to
  report the resolved path):

  ```bash
  # from the target node, pulling off the old node (pistachio)
  rsync -avP pistachio:/local_nvme1/mborjigi/eagle-drafters/sql-cold-e3-slim/ \
      $STORE/eagle-drafters/sql-cold-e3-slim/
  python $FASTRL/eagle-train/scripts/slim_drafter.py rehydrate \
      --slim  $STORE/eagle-drafters/sql-cold-e3-slim \
      --target Qwen/Qwen2.5-Coder-7B-Instruct \
      --out   $STORE/eagle-drafters/sql-cold-e3
  ```
  Then point `SPEC_MODEL_PATH` at the rehydrated dir. Node seen 2026-08-20:
  Oscar, `/oscar/scratch/mborjigi/{fastrl,granular-cais-rl}`.
* `/local_nvme1/mborjigi/data/text2sql-data/{bench512,bench650}.parquet` and
  `fastrl/data/skyrl_sql/*.parquet` — cheap to rebuild, see
  [[project_engine_ab_sglang_vs_vllm]] for the filter (prompts <= 4096).
* Traces: `fastrl/output/engine_ab3/trace.*.jsonl`,
  `granular/output/engine_ab_trace/*.jsonl` — keep if you want to re-replay
  without re-recording.
* `/local_nvme1/mborjigi/eagle-ckpt/` (deepspeed ckpts),
  `/local_nvme1/mborjigi/eagle-cache/` (70 GB hidden states) — regenerable,
  probably skip.
* Both `.venv`s and the HF cache `/local_nvme0/mborjigi/hf` — rebuild on the new
  node (`uv sync --inexact` for granular; `eagle-train/install_uv.sh` for the
  drafter env). See [[project_fastrl_install]] for the flashinfer wheel patch and
  [[ops_venv_cross_contamination]].

## Operational gotchas that cost time here

* **Launch and wait in SEPARATE Bash calls.** Combining `setsid nohup ... &`
  with the `until` wait loop in one backgrounded call means killing the task
  kills the run. Two runs lost this way.
* **`pkill -f <pattern>` matches its own shell** when the pattern appears in the
  command line — it SIGTERMs itself (exit 144) and the rest of the command never
  runs. Use a bracket class: `pgrep -f "[s]glang::"`. Cost 2 more runs.
* Killed SD runs leave `sglang::detokenizer` zombies squatting on port 34257 ->
  next run dies with `EADDRINUSE`. Reap before relaunching.
* `ray stop --force` will try (and fail) to kill collaborators' GCS servers —
  harmless, but don't chase it.

---

### UPDATE 2026-08-31: ROOT CAUSE FOUND AND FIXED. It was our patch, not FastRL. A double `prepare_for_decode` on the adaptive-downgrade path was injecting stale KV into every decode step.

`### UPDATE 2026-08-30b` narrowed this to "cross-request state corruption in the
tree/KV plumbing" and listed a standalone-sglang control as the next step. That
control was never needed: assuming upstream FastRL is correct and auditing only
our own diff found it immediately.

**The bug.** Commit `0f7c378` (our fix for the adaptive-SD multi-turn crash) added
this to `EAGLEWorker.forward_batch_generation`'s `if not enable_sd:` branch in
`third-party/sglang/python/sglang/srt/speculative/eagle_worker.py`:

```python
batch.spec_algorithm = SpeculativeAlgorithm.NONE
batch.spec_info = None
batch.output_ids = torch.tensor([req.output_ids[-1] for req in batch.reqs], ...)
batch.prepare_for_decode(skip_prepare=False)
```

Its comment justified this by claiming `skip_prepare` "is decided from the STATIC,
engine-level spec_algorithm". **It is not.** It is decided from the *per-batch*
field, and upstream already handles this case in `Scheduler.update_running_batch`
(`scheduler.py:2229-2238`):

```python
if (self.adaptive_spec_threshold is not None
        and self.adaptive_spec_threshold > 0
        and batch.batch_size() > self.adaptive_spec_threshold):
    self.running_batch.spec_algorithm = SpeculativeAlgorithm.NONE
batch.prepare_for_decode(skip_prepare=not self.running_batch.spec_algorithm.is_none())
```

Upstream sets `spec_algorithm = NONE` *precisely so that* `skip_prepare` becomes
False and the real preparation runs. Our patch then ran it a **second time** in the
same decode round. (Confirmed by diff that this scheduler block is upstream; our
only edit to `scheduler.py` is the `SGLANG_TOLERATE_KV_LEAK` hatch.)

**Why that corrupts output.** `prepare_for_decode`'s body calls
`alloc_for_decode`, which allocates one fresh KV slot per request, writes it into
`req_to_token_pool` at `locs = seq_lens`, and then does `seq_lens.add_(1)`. Run
twice per round:

- call 1 allocates slot **A**, maps it at position `L`, `seq_lens: L -> L+1`
- call 2 allocates slot **B**, maps it at `L+1`, `seq_lens: L+1 -> L+2`
- the forward pass writes this step's token KV to `out_cache_loc` = **B**

Slot **A** is therefore allocated, mapped into the request's token table, inside
`seq_lens`, **never written, and never freed**. Every decode step splices one
phantom token of whatever stale KV the allocator last left there -- typically
another request's freed KV -- into the attention context, and `seq_lens` advances
2 per generated token.

**Why it only bit at production scale.** The downgrade path is taken only when
`batch_size > bs_threshold` (default 32). At 256x5 there are ~1280 concurrent
requests, so it fires on nearly every decode step; at smoke scale (8x5) the batch
drops below 32, speculation engages properly, and the normal path is clean. That
is why acceptance looked healthy (2.05-8.50) at smoke scale while production
collapsed totally. It also retires the "regression after `75df8ae`" suspicion:
`0f7c378` predates `75df8ae`, and `75df8ae` is simply what first made
production-scale batches runnable at all. The bug was latent, not new.

**The fix.** Read `batch.spec_algorithm.is_none()` *before* clobbering the field --
it is an exact predicate for "the scheduler already prepared this batch this
round", since nothing ever restores that field -- and skip the redundant
preparation. The manual preparation is kept only for the case upstream genuinely
does not cover: `batch_size <= threshold`, reached via the warmup /
`pending_transition_to_sd` branches of `should_enable_sd()`, which is the
`batch_size=20` crash `0f7c378` was written for.

**Measured, greedy, single-turn, 2 GPUs, 128x5, `gpu_memory_utilization=0.45` --
identical config in all three arms, only the code differs:**

| arm | non-ASCII chars | per 1k chars | mean resp chars | `</sql>` never closed | uniq of 5 |
|---|---|---|---|---|---|
| SD-off | 10 | 0.01 | 1098 | 3.4% | 1.49 |
| SD-on, **pre-fix** | 102,415 | 27.96 | 5722 | **88.6%** | 4.69 |
| SD-on, **post-fix** | 10 | 0.01 | 1156 | 5.0% | 1.73 |

The pre-fix arm was re-run at the *new* config specifically to rule out the
GPU-count/batch change as the explanation. It reproduces the corruption there, so
the attribution is to the code, not the config.

**Bitwise (greedy), same config:**

| comparison | exact match |
|---|---|
| SD-off vs SD-off (two runs) | **128/128 = 1.000** |
| SD-on vs SD-on (two runs) | 127/128 = 0.992 |
| SD-on vs SD-off, pre-fix | 0/256 = 0.000 (median divergence char 6) |
| SD-on vs SD-off, post-fix | 56/128 = 0.438 (median divergence char 154) |

**On the residual 0.438 -- stated honestly.** The engine is run-to-run
deterministic (SD-off vs SD-off is exactly 1.000), so 1.000 is the real ceiling and
post-fix SD is *not* bit-identical to non-SD. What the evidence does show is that
the residual is a different phenomenon from the corruption: junk tokens are gone
entirely (10, exactly the SD-off count), mean length matches (1156 vs 1098 vs the
pre-fix 5722), the divergences are coherent alternative continuations rather than
spliced garbage, and the fixed SD path is itself deterministic at 0.992. The
natural explanation is floating-point: the verify step runs the target model over
several candidate tokens at once, so kernel shapes and reduction order differ from
single-token decode and argmax flips on near-ties. This engine is *demonstrably*
that sensitive -- with SD off entirely, the same prompt repeated 5 times in one
batch already yields 1.49 distinct greedy outputs, purely from batch position. But
this is an inference from consistent evidence, **not** a proof that every residual
divergence is benign numerics, and it should not be recorded as one.

**Corrections to the earlier record.** Three hypotheses in `UPDATE 2026-08-30b`
were wrong and their diagnostics have been removed: the uninitialised `predict`
buffer (`eagle_info.py` is now byte-identical to upstream again), the
`context_length` cap (its `FASTRL_ENGINE_CONTEXT_LEN` override is gone; the comment
now records the refutation), and, from further back, "SD destroys GRPO diversity".
The elimination list in `UPDATE 2026-08-30b` was sound as far as it went -- every
item on it really was eliminated -- but all of it searched sglang, and the defect
was in our own patch to sglang the whole time.

**Still open.** The 2-token `token_to_kv_pool` accounting leak is very likely the
same defect (slot **A** is leaked once per request per downgraded decode step), and
`SGLANG_TOLERATE_KV_LEAK` should now be unnecessary -- but this is **unverified**:
the leak only surfaced at step 6 of a training run and no training run has been made
since. The hatch stays until a training run confirms it. The obvious next step is to
re-run the 256x5 single-turn SD on/off training ablation, which pre-fix collapsed
256/256 groups on all 5 steps and never ran a single optimizer step; that is the
number that decides whether FastRL's SD actually delivers.
