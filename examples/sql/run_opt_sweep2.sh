#!/usr/bin/env bash
# Sweep 2, after sweep 1 ruled out the training-side levers.
#
# What sweep 1 established (all at the fixed 256x5 workload):
#   * GRAD_CKPT=False OOMs in update_actor -- d1/d2/d3 all died there with
#     41.4-41.9 GiB allocated of 44.4 GiB. The baseline is already at the
#     memory wall, so trading memory for compute is not available.
#   * PARAM_OFFLOAD=False at GPU_MEM_UTIL=0.4 kills sglang at init
#     ("max_total_num_tokens <= 0") -- resident FSDP params starve the KV pool.
#
# What is left is the rollout side. At gpu_mem_util=0.4 the engine gets
# 0.4*44.4 - 15.2(weights) = 2.6GB of KV = ~45k tokens, against ~3.2M tokens of
# demand per step (1280 seqs x ~2500 tok). That is a severe queueing constraint
# on the 135s gen phase, and the engine sleeps during training, so a larger
# fraction should not cost training memory.
#
# Workload params (256 x 5, lr, temperature) stay fixed throughout.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VIRTUAL_ENV="${FASTRL_VENV:-$PWD/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

OUT=output/opt_sweep
mkdir -p "$OUT"
STEPS=${STEPS:-3}

run_cfg() {
    local name="$1"; shift
    local log="$OUT/${name}.log"
    if [[ -f "$log" && -z "${FORCE:-}" ]]; then
        echo "== skip $name (log exists)"; return 0
    fi
    echo "== $name : $* "
    timeout -k 30 "${PER_CFG_TIMEOUT:-2700}" \
      env "$@" \
        MAX_STEPS="$STEPS" \
        EXPERIMENT_NAME="opt-$name" \
        SKYRL_SQL_DUMP_PATH="" \
        bash examples/grpo_sql_baseline.sh >"$log" 2>&1
    local rc=$?
    pkill -f main_fastrl >/dev/null 2>&1 || true
    pkill -f "sglang::scheduler" >/dev/null 2>&1 || true
    "$VIRTUAL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
    sleep 15
    if ! grep -q "timing_s/update_actor" "$log"; then
        echo "== $name FAILED (rc=$rc, no completed step)"; return 1
    fi
    echo "== $name ok (rc=$rc)"
    return 0
}

# E1/E2: give the engine a real KV cache. Everything else is the baseline.
run_cfg e1_mem06 SPECULATIVE=false GPU_MEM_UTIL=0.6
run_cfg e2_mem07 SPECULATIVE=false GPU_MEM_UTIL=0.7

# E3: pick the better fraction, then shave the weight-sync/wake overhead that
# shows up as the 17s `reshard` inside gen.
BEST_MEM=0.6
if [[ -f "$OUT/e2_mem07.log" ]] && grep -q "timing_s/update_actor" "$OUT/e2_mem07.log"; then
    BEST_MEM=0.7
fi
echo "== using GPU_MEM_UTIL=$BEST_MEM for e3/e4"
run_cfg e3_sync SPECULATIVE=false GPU_MEM_UTIL=$BEST_MEM \
    WSYNC_BUCKET=2048 MULTI_STAGE_WAKE=True FWD_PREFETCH=True

# E4: speculative decoding on top of the best throughput config, with the
# instruct-lineage drafter (the only one that was ever faster than SD-off).
# Its MHA draft-KV is ~7x heavier than GQA, so it gets one step less headroom.
run_cfg e4_sd SPECULATIVE=true GPU_MEM_UTIL=0.5 \
    SPEC_MODEL_PATH=Tengyunw/qwen_2.5_7b_instruct_eagle2_v0 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

echo "== sweep2 done"
