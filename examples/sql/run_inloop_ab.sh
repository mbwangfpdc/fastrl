#!/usr/bin/env bash
# Phase 4 (in-loop): does the adapted drafter help a real 256x5 RL step?
# Offline it reaches ~2.9 accept length vs ~1.18 for both off-the-shelf drafters.
# Our drafter is GQA (kv_heads 4, inherited from the target config), so its draft
# KV is ~1/7th of Tengyunw's MHA and it fits at gpu_mem_util 0.6 -- the same
# fraction as the SD-off reference (e1_mem06, step 1 = 1399s), making this a
# matched comparison rather than one confounded by engine memory.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VIRTUAL_ENV="$PWD/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
OUT=output/opt_sweep; mkdir -p $OUT
name=h1_sd_ourdrafter
timeout -k 30 4200 env SPECULATIVE=true GPU_MEM_UTIL=0.6 \
    SPEC_MODEL_PATH=/local_nvme1/mborjigi/eagle-drafters/sql-cold-e3 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    MAX_STEPS=2 EXPERIMENT_NAME="opt-$name" SKYRL_SQL_DUMP_PATH="" \
    bash examples/grpo_sql_baseline.sh > "$OUT/$name.log" 2>&1
echo "rc=$?"
pkill -9 -f main_fastrl >/dev/null 2>&1 || true
pkill -9 -f "sglang::scheduler" >/dev/null 2>&1 || true
"$VIRTUAL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
echo "steps: $(grep -o 'timing_s/step:[0-9.]*' "$OUT/$name.log" | tr '\n' ' ')"
echo "gen  : $(grep -o 'timing_s/generate_sequences:[0-9.]*' "$OUT/$name.log" | tr '\n' ' ')"
grep -qi "OutOfMemory" "$OUT/$name.log" && echo "-> OOM"
