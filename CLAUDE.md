# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this checkout is

A **local fork of MIT-Han-Lab's FastRL** (verl + a vendored sglang with adaptive
EAGLE speculative decoding, "TLT" in their paper), used as an external baseline
to compare against `granular-cais-rl` on the SkyRL-SQL workload. Local
modifications are ours and are not intended for upstream. `origin` is
`mbwangfpdc/fastrl` (a personal fork), so pushing here is fine.

## Active work / handoff

**[RESUME_TLT_DRAFTER.md](RESUME_TLT_DRAFTER.md) — read this first.** What is
done (the sglang-vs-vLLM engine A/B: vLLM ~7% faster on a bit-identical
workload, so the 2.6x system gap is framework not engine), what is half-finished
(four fixes making TLT's opportunistic drafter training reachable — one
confirmed, three unvalidated), the blocker (FastRL's speculative decoding
crashes on the multi-turn path whenever speculation actually engages), the exact
resume command, and how to move the trained drafter between nodes without a ~2h
retrain.

## Running it

`examples/grpo_sql_baseline.sh` is the entry point; every execution knob is an
env override. See `examples/sql/README.md` for the tuning results at the real
256x5 batch and the head-to-head numbers, and
`eagle-train/SQL_DRAFTER_RESULTS.md` for the drafter adaptation.

Always set `VIRTUAL_ENV="$PWD/.venv"` — shells here export granular-cais-rl's
venv, and an unqualified `uv pip install` will land in the wrong one and can
downgrade torch under a live run.
