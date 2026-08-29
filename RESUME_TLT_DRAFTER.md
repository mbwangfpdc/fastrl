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
