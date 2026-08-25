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
(`0f7c378`). On Oscar: with the paper's matched target+drafter pair and
Adaptive Drafter training OFF, the Adaptive Rollout Engine gets **real
acceptance (2.05-8.50)** — the old "accept len 1.00 / SD is pure overhead"
result was the wrong-drafter artifact, not a real limitation
(`run_tlt_flagship_norafter_slurm.sh`, job 5242734). With Adaptive Drafter
training also on, the same smoke config instead **hangs the whole node**
(job 5197256: 12h+ at 0% GPU util, node process table exhausted) — worse than
the earlier "contributes nothing" verdict, and still unresolved. See
`RESUME_TLT_DRAFTER.md`'s `### UPDATE 2026-08-25 (Oscar)` for the full
writeup and next steps.

## Running it

`examples/grpo_sql_baseline.sh` is the entry point; every execution knob is an
env override. See `examples/sql/README.md` for the tuning results at the real
256x5 batch and the head-to-head numbers, and
`eagle-train/SQL_DRAFTER_RESULTS.md` for the drafter adaptation.

Always set `VIRTUAL_ENV="$PWD/.venv"` — shells here export granular-cais-rl's
venv, and an unqualified `uv pip install` will land in the wrong one and can
downgrade torch under a live run.
