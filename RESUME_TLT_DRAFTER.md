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

## Where this stands

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

## THE BLOCKER: FastRL's SD crashes on multi-turn when speculation engages

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

## How to resume

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
