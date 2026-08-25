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
(`0f7c378`); drafter training has now been **observed running end-to-end** in
multi-turn. The live blocker is that with the paper's own matched
target+drafter pair, speculation accepts **nothing** (`accept len 1.00` on every
decode batch), making SD pure overhead — ranked hypotheses and the non-RL
control that discriminates between them are in that file. Work is moving to
Slurm: `scripts/run_tlt_flagship_slurm.sh` (run with `SMOKE=1` first).

## Running it

`examples/grpo_sql_baseline.sh` is the entry point; every execution knob is an
env override. See `examples/sql/README.md` for the tuning results at the real
256x5 batch and the head-to-head numbers, and
`eagle-train/SQL_DRAFTER_RESULTS.md` for the drafter adaptation.

Always set `VIRTUAL_ENV="$PWD/.venv"` — shells here export granular-cais-rl's
venv, and an unqualified `uv pip install` will land in the wrong one and can
downgrade torch under a live run.
