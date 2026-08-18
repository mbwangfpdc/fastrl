#!/usr/bin/env bash
# Sweep 3 -- the final two configurations, at the empirically-best engine
# fraction.
#
# Established so far (all at the fixed 256x5 workload, 4x L40S 44.4GiB):
#   * Training side is closed. GRAD_CKPT=False OOMs in update_actor at
#     41.4-41.9 GiB of 44.4; PARAM_OFFLOAD=False starves sglang's KV pool at
#     init. Offload is load-bearing here, not a slow default.
#   * gpu_memory_utilization has a narrow optimum and a cliff:
#       0.4 -> step 544/480    0.5 -> step 531/469    0.6 -> step 1399
#     0.6 does not just fail to help, it triples the TRAINING phases
#     (old_log_prob 89->281s, update_actor 294->942s), i.e. the engine does not
#     hand its memory back cleanly on sleep and the trainer thrashes.
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
    pkill -9 -f main_fastrl >/dev/null 2>&1 || true
    pkill -9 -f "sglang::scheduler" >/dev/null 2>&1 || true
    "$VIRTUAL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
    sleep 20
    if ! grep -q "timing_s/update_actor" "$log"; then
        echo "== $name FAILED (rc=$rc, no completed step)"; return 1
    fi
    echo "== $name ok (rc=$rc)"
    return 0
}

# F1: best engine fraction + the remaining low-risk overhead knobs. Bigger
# weight-sync buckets and multi-stage wake both target the ~17-19s `reshard`
# that sits inside gen; forward_prefetch overlaps the next all-gather.
run_cfg f1_tuned SPECULATIVE=false GPU_MEM_UTIL=0.5 \
    WSYNC_BUCKET=2048 MULTI_STAGE_WAKE=True FWD_PREFETCH=True

# F2: speculative decoding on top, with the instruct-lineage drafter -- the only
# one that ever beat SD-off. Its MHA draft-KV is ~7x heavier than GQA, hence the
# same 0.5 fraction rather than more.
run_cfg f2_sd SPECULATIVE=true GPU_MEM_UTIL=0.5 \
    SPEC_MODEL_PATH=Tengyunw/qwen_2.5_7b_instruct_eagle2_v0 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

echo "== sweep3 done"
