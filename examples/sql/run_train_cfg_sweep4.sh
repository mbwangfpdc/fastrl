#!/usr/bin/env bash
# Confirmation round. Two independent wins showed up on top of SP2:
#   forward_prefetch=True   0.208 ms/tok   (vs 0.249 baseline)
#   token_budget 8192       0.233 ms/tok
# Combine them, and repeat each single-knob config once, because `gen` varied
# 2.5x across configs that cannot affect generation -- so the noise floor here
# is not yet known and a 15% difference from n=1 may not be real.
set -u
export FASTRL_ROOT=/local_nvme1/mborjigi/fastrl
source "$FASTRL_ROOT/examples/sql/cfg_probe_lib.sh"
OUT=$FASTRL_ROOT/output/train_cfg_sweep
mkdir -p "$OUT"
STEPS=2; MAXSEC=1500; STALL=240
COMMON=(NGPUS=4 TP=1 TRAIN_PROMPT_BSZ=32 MINI_BSZ=32 N_RESP=5 MAX_STEPS=$STEPS
        SPECULATIVE=false
        TRAIN_FILE=data/skyrl_sql/bench650.parquet
        VAL_FILE=data/skyrl_sql/validation.parquet)
probe () { local n=$1; shift; probe_run "$n" "$OUT" "$STEPS" "$MAXSEC" "$STALL" -- \
             "${COMMON[@]}" EXPERIMENT_NAME="$n" "$@"; }

probe M_pref_b8k    ULYSSES=2 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.6 FWD_PREFETCH=True
probe M_pref_b8k_r2 ULYSSES=2 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.6 FWD_PREFETCH=True
probe H_prefetch_r2 ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 FWD_PREFETCH=True
probe I_budget8k_r2 ULYSSES=2 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.6
echo "### done4"
