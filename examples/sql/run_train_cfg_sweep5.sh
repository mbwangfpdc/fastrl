#!/usr/bin/env bash
# The optimizer-residency question, tested properly this time.
#
# The earlier C2/C3 probes conflated two variables: they turned OPTIM_OFFLOAD
# off AND dropped mem_fraction_static to 0.30/0.20 "to make room". That is
# backwards for sglang, whose pool is
#     KV = available_after_weights - total * (1 - mem_fraction_static)
# so a LOWER fraction means a LARGER reserve subtracted and a SMALLER pool. The
# failures at 0.30/0.20 are consistent with the fraction alone and say nothing
# about the optimizer.
#
# The clean test is to hold the fraction at the known-good 0.6 and flip only the
# offload. That was config E in the first sweep, which failed for an unrelated
# reason (the drop_zero_signal alignment bug, since fixed), so it has never
# actually been measured.
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

probe E2_optres_m060   ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 OPTIM_OFFLOAD=False
probe E3_nooff_m060    ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 PARAM_OFFLOAD=False OPTIM_OFFLOAD=False
echo "### done5"
