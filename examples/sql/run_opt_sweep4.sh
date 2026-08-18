#!/usr/bin/env bash
# Sweep 4 -- a CONTROL, then the speculative-decoding config.
#
# Why a control. Two sweep configs (e1 at mem_util 0.6, f1 at mem_util 0.5 with
# wake/bucket/prefetch tuning) both regressed to an almost identical signature:
#   old_log_prob 89 -> ~281s,  update_actor ~294 -> ~950s,  step ~530 -> ~1400-1500s
# Both were the FIRST config in a freshly-launched sweep, whereas both healthy
# reference numbers (train10.log, sdoff_05.log) came from directly-launched
# runs. With param/optimizer offload on, Adam moments live in HOST RAM, so a 3x
# training slowdown is equally consistent with host-memory pressure as with the
# knob under test. That is a confound, not a result.
#
# c0_control re-runs the plain mem_util=0.5 baseline -- no other changes --
# through this same driver. If it reproduces ~530/469s, the driver is clean and
# the e1/f1 regressions are attributable to their knobs. If it comes back
# ~1500s, every sweep conclusion here is invalid and the knobs are exonerated.
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
    echo "-- host mem before: $(free -g | awk '/^Mem:/{print $7" GB available"}')"
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
    echo "-- host mem after:  $(free -g | awk '/^Mem:/{print $7" GB available"}')"
    if ! grep -q "timing_s/update_actor" "$log"; then
        echo "== $name FAILED (rc=$rc, no completed step)"; return 1
    fi
    echo "== $name ok (rc=$rc)"
    return 0
}

# The control: identical to the healthy sdoff_05 reference, run via this driver.
run_cfg c0_control SPECULATIVE=false GPU_MEM_UTIL=0.5

# Speculative decoding on the best known-good configuration.
run_cfg f2_sd SPECULATIVE=true GPU_MEM_UTIL=0.5 \
    SPEC_MODEL_PATH=Tengyunw/qwen_2.5_7b_instruct_eagle2_v0 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

echo "== sweep4 done"
