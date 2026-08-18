#!/usr/bin/env bash
# Sweep 5 -- the corrected sweep, at the real 256x5 workload.
#
# WHAT WENT WRONG IN SWEEPS 1-4: every reference run used to judge them
# (train10.log, sdoff_05.log, the SD A/Bs) was collected at train_batch_size=64,
# while the sweep configs ran at 256. The apparent "3x training slowdown" and
# the "cliff at gpu_mem_util 0.6" were both artifacts of comparing a 256-batch
# run against a 64-batch reference. Corrected picture, all at batch 256, step 1:
#     mem_util 0.5 -> 1536s   (c0_control)
#     mem_util 0.5 + tuning -> 1516s   (f1_tuned)
#     mem_util 0.6 -> 1399s   (e1_mem06)
# i.e. the HIGHER engine fraction wins at this batch size, matching the KV-cache
# arithmetic: at 256x5 the engine serves ~1280 concurrent requests, so KV
# capacity is the binding constraint on generation.
#
# Goal per the brief: a REASONABLE, fair-shot configuration for FastRL on this
# hardware -- not a hunt for the global optimum. Two configs only.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VIRTUAL_ENV="${FASTRL_VENV:-$PWD/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

OUT=output/opt_sweep
mkdir -p "$OUT"
# 2 steps: step 1 plus one steady-state step. At ~1400s/step a 3-step run costs
# 70min per config, which is not worth it for a fair-shot number.
STEPS=${STEPS:-2}

run_cfg() {
    local name="$1"; shift
    local log="$OUT/${name}.log"
    if [[ -f "$log" && -z "${FORCE:-}" ]]; then
        echo "== skip $name (log exists)"; return 0
    fi
    echo "== $name : $* "
    timeout -k 30 "${PER_CFG_TIMEOUT:-4200}" \
      env "$@" \
        MAX_STEPS="$STEPS" \
        EXPERIMENT_NAME="opt-$name" \
        SKYRL_SQL_DUMP_PATH="" \
        bash examples/grpo_sql_baseline.sh >"$log" 2>&1
    local rc=$?
    pkill -9 -f main_fastrl >/dev/null 2>&1 || true
    pkill -9 -f "sglang::scheduler" >/dev/null 2>&1 || true
    "$VIRTUAL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
    sleep 20
    if ! grep -q "timing_s/update_actor" "$log"; then
        echo "== $name FAILED (rc=$rc, no completed step)"; return 1
    fi
    echo "== $name ok (rc=$rc): $(grep -o 'timing_s/step:[0-9.]*' "$log" | tr '\n' ' ')"
    return 0
}

# G1: push the engine fraction one notch past the current best (0.6 -> 0.7).
# Training peaks near 42GiB of 44.4, so this is also the point where the engine
# and the trainer stop fitting; if it regresses, 0.6 is the answer.
run_cfg g1_mem07_sdoff SPECULATIVE=false GPU_MEM_UTIL=0.7

# G2: speculative decoding a fair shot at the real batch size, with the
# instruct-lineage drafter (the only one that ever beat SD-off). NB: at 256x5
# the running batch sits far above speculative.bs_threshold=32 for most of the
# rollout, so adaptive SD only engages on the tail by construction.
run_cfg g2_mem06_sd SPECULATIVE=true GPU_MEM_UTIL=0.6 \
    SPEC_MODEL_PATH=Tengyunw/qwen_2.5_7b_instruct_eagle2_v0 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

echo "== sweep5 done"
