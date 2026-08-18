#!/usr/bin/env bash
# Sweep 6 -- speculative decoding, matched against the SD-off control.
# g2 OOMed at gpu_mem_util 0.6: the Tengyunw drafter uses full MHA (kv_heads 28
# vs 4 for GQA), so its draft KV is ~7x heavier and the engine hit 35.96GiB.
# 0.5 is the fraction this drafter fits in -- and it matches c0_control
# (0.5, SD off, step 1 = 1536s), so this is a clean A/B at the real 256x5 batch.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VIRTUAL_ENV="${FASTRL_VENV:-$PWD/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
OUT=output/opt_sweep; mkdir -p "$OUT"
name=g3_mem05_sd
log="$OUT/${name}.log"
echo "== $name"
timeout -k 30 4200 env SPECULATIVE=true GPU_MEM_UTIL=0.5 \
    SPEC_MODEL_PATH=Tengyunw/qwen_2.5_7b_instruct_eagle2_v0 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    MAX_STEPS=2 EXPERIMENT_NAME="opt-$name" SKYRL_SQL_DUMP_PATH="" \
    bash examples/grpo_sql_baseline.sh >"$log" 2>&1
rc=$?
pkill -9 -f main_fastrl >/dev/null 2>&1 || true
pkill -9 -f "sglang::scheduler" >/dev/null 2>&1 || true
"$VIRTUAL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
echo "== $name rc=$rc: $(grep -o 'timing_s/step:[0-9.]*' "$log" | tr '\n' ' ')"
