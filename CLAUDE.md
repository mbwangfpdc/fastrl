# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this checkout is

A **local fork of MIT-Han-Lab's FastRL** (verl + a vendored sglang with adaptive
EAGLE speculative decoding, "TLT" in their paper), used as an external baseline
to compare against `granular-cais-rl` on the SkyRL-SQL workload. Local
modifications are ours and are not intended for upstream. `origin` is
`mbwangfpdc/fastrl` (a personal fork), so pushing here is fine.

## Active work / handoff

**[RESUME_TLT_DRAFTER.md](RESUME_TLT_DRAFTER.md) — read this first**, starting
at `## STATUS 2026-08-25`, which supersedes several older sections of that file.

Short version: the SD-off 35-step e2e run **completed** (reward -0.168 -> 0.562;
merged HF checkpoint on pistachio); the old multi-turn SD crash is **fixed**
(`0f7c378`). With the paper's matched target+drafter pair and Adaptive
Drafter training OFF, the Adaptive Rollout Engine gets **real acceptance**
(2.05-8.50 at smoke scale) — the old "accept len 1.00 / SD is pure overhead"
result was the wrong-drafter artifact, not a real limitation. A full clean
10-step production-batch (256x5) run **completed 2026-08-26** after fixing
three bugs (an EAGLE buffer sized off the model's unused 128K native context
instead of the actual 8K working ceiling; a related SD admission-check
margin; a wrong Hydra path for the NCCL collective timeout) — see
`sglang_rollout.py`'s `AsyncEngine(...)` call site and
`scripts/run_tlt_norafter_prod_slurm.sh`'s header for the fixes. Result:
**reward stayed flat (-0.75 to -0.80) across all 10 steps** and **252/256
GRPO groups were dropped as zero-signal every step** — the paper's base
(non-instruct) `Qwen2.5-7B` basically doesn't learn on this SQL workload in
this window, unlike the Coder-Instruct target used elsewhere in this
comparison. With Adaptive Drafter training also on, the flagship smoke config
still **hangs the whole node** (job 5197256: 12h+ at 0% GPU util, node
process table exhausted) — unresolved, not yet revisited.

**2026-08-29**: the "252/256 zero-signal groups" finding above is confirmed
real (clean SD on/off ablation, same model, reward -0.80 flat with SD vs
+0.45 climbing without). Found and fixed a real, separate bug along the way:
single-turn generation (`multi_turn.enable=False`) hung on every run, SD on
or off, because `_generate_with_drafter` released GPU memory unconditionally
instead of gating it like its multi-turn sibling does — see that function in
`sglang_rollout.py`. Whether the diversity collapse itself is specific to
multi-turn is **still open**: a 16-group single-turn smoke test showed SD-on
and SD-off collapsing at the same rate (unlike multi-turn's stark gap), but
that sample size doesn't yet distinguish "not multi-turn-specific" from "too
small to see it." See `RESUME_TLT_DRAFTER.md`'s `### UPDATE 2026-08-26` and
`### UPDATE 2026-08-29` for the full writeups.

**2026-08-30 — that open question is now answered, and the explanation for it
changes.** Read `### UPDATE 2026-08-30` in `RESUME_TLT_DRAFTER.md`; it
supersedes the "diversity collapse" language above. The single-turn 256x5 pair
(`scripts/run_tlt_singleturn_prod_local.sh`, one script parameterised by
`SPECULATIVE`) reproduces the collapse, so it is **not** multi-turn-specific:
SD-off drops 236 -> 188 of 256 groups and starts learning, SD-on drops
**256/256 on all 5 steps and never runs a single optimizer step**, at 2.44x
the step time — and with speculation working *normally* (mean accept len
2.15, max 5.52), so this is not the old wrong-drafter artifact. A 2x2
rollout-only probe over the trajectory text
(`scripts/run_tlt_diversity_probe.sh` + `scripts/analyze_group_diversity.py`)
gives the mechanism: **enabling SD makes responses ~3.3x longer and takes
`</sql>` closure from 6-7% missing to 91-94% missing**, which in multi-turn
severs the agent loop (3.08 -> 0.97 turns, `<solution>` 54% -> 2%) so every
rollout hits the -1.0 format-error path and every group ties. Within-group
diversity is *perfect* in all four cells (`exact_dup_rate` 0.000, 4.98-5.00
unique of 5) — it is a termination/format collapse, not a diversity collapse.
Two smaller results worth knowing: a missing per-step metrics line is not
cosmetic, it is exactly the signature of a 256/256 step that skipped its
update (grep `every group was zero-signal`); and a 2-token KV-pool accounting
leak kills the scheduler on the single-turn path at step 6, worked around by
opt-in `SGLANG_TOLERATE_KV_LEAK=1`, still unfixed.

**2026-08-30b — root cause narrowed; see `### UPDATE 2026-08-30b`.** SD is
**injecting junk tokens**, not shifting a distribution: greedy SD-on output
has 23,048 non-ASCII fragments vs SD-off's 2, spliced into otherwise-coherent
text (`<think>ük>First teh, I will checkük if the 'ükApartükmentsük' table`).
Under greedy — where correct SD is *bit-identical*, not merely
distribution-preserving — SD-on still misses `</sql>` 91.1% of the time vs
3.9%, and returns 4.66 different answers per 5 identical prompts (SD-off:
1.50), so the defect is **nondeterministic and batch/state-dependent**.
Eliminated with evidence: the stochastic accept kernel (greedy uses a
different one and still corrupts), `enforce_eager`, an uninitialised `predict`
buffer (sentinel test), the `context_length` cap, relaxed accept thresholds,
the greedy fallback, verify-path sampling params, and a stale drafter
embed/lm_head. Next: a standalone sglang control with no verl, then a
concurrency sweep — prime suspect is cross-request state corruption in the
tree/KV plumbing, which would also explain the allocator leak.

## Running it

`examples/grpo_sql_baseline.sh` is the entry point; every execution knob is an
env override. See `examples/sql/README.md` for the tuning results at the real
256x5 batch and the head-to-head numbers, and
`eagle-train/SQL_DRAFTER_RESULTS.md` for the drafter adaptation.

Always set `VIRTUAL_ENV="$PWD/.venv"` — shells here export granular-cais-rl's
venv, and an unqualified `uv pip install` will land in the wrong one and can
downgrade torch under a live run.
