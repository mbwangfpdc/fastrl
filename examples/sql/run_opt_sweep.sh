#!/usr/bin/env bash
# Throughput sweep for the "maximum performance by configuration" baseline.
#
# Rationale: at our defaults a 445s step decomposes as
#   update_actor 235s (53%) | old_log_prob 75s (17%) | gen 135s (30%)
# so the levers that matter are on the TRAINING side. Speculative decoding can
# only address the ~116s of generate_sequences (26% of the step), and enabling
# it also forces `disable_overlap_schedule=True` and `cuda_graph_max_bs=32`
# engine-wide (see sglang_rollout.py speculative_args), so it is tested last and
# only on top of the best training config.
#
# Every knob varied here changes microbatch splitting / memory placement only.
# None of them change the gradient, so all configs remain comparable to the
# granular-cais-rl baseline. Workload params (batch, n, lr, temperature) are
# held fixed on purpose.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Pin the venv here too: shells on this box export granular-cais-rl's
# VIRTUAL_ENV, so a bare `ray` between configs would drive the WRONG cluster.
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
    # Hard cap: 3 steps should never exceed ~30min. A worker that dies during
    # sglang init leaves the driver blocked in Ray forever, which stalled the
    # whole sweep once already -- so bound it rather than trust a clean exit.
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
    sleep 10
    # A config that never reached a step is a failure even if the driver exits 0.
    if ! grep -q "timing_s/update_actor" "$log"; then
        echo "== $name FAILED (rc=$rc, no completed step)"; return 1
    fi
    echo "== $name ok (rc=$rc)"
    return 0
}

# Ordering rationale. Back-of-envelope from the baseline: old_log_prob is a pure
# forward over ~3.2M tokens in 75s => ~600 TFLOP/s aggregate across 4 L40S, i.e.
# these phases are already near roofline rather than stalled on data movement.
# That makes gradient checkpointing (8NT of compute vs 6NT without) the largest
# *real* lever, and CPU offload a minor one -- so test the big lever first, on
# the known-good memory configuration.
#
# NOTE: c1_nooffload (PARAM_OFFLOAD=False at GPU_MEM_UTIL=0.4) already failed --
# resident FSDP params starve sglang's KV pool and init dies with
# "max_total_num_tokens <= 0". Any no-offload config must also raise
# GPU_MEM_UTIL, which is why d3 below pairs them.

# D1: drop gradient checkpointing -- buys back the recomputed forward (~25% of
# update_actor by the roofline estimate).
run_cfg d1_nockpt SPECULATIVE=false GRAD_CKPT=False

# D2: + drop Ulysses SP (per-layer all-gather/reduce-scatter is pure overhead
# once the token budget alone clears the max_seq_len assert).
run_cfg d2_nockpt_nosp SPECULATIVE=false GRAD_CKPT=False ULYSSES=1 TOKEN_BUDGET=8192

# D3: + keep params/optimizer resident. Needs the higher engine fraction, since
# sglang sizes its KV pool from memory that is actually free at init.
run_cfg d3_nooffload SPECULATIVE=false GRAD_CKPT=False ULYSSES=1 TOKEN_BUDGET=8192 \
    PARAM_OFFLOAD=False OPTIM_OFFLOAD=False GPU_MEM_UTIL=0.6

echo "== sweep done"
