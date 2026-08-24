#!/usr/bin/env bash
# Throughput probes for the 35-step e2e config, run under the fail-fast watchdog.
#
# B (SP1, budget 8192, offloads on) is the known-good reference at 32x5:
#   update_actor 127.3  old_log_prob 42.7  gen 40.6  step 232.5
# Everything here tries to beat its update_actor without losing the fit.
#
# Peak training memory is set by ppo_max_token_len_per_gpu (the microbatch), not
# by the prompt batch, so probing at 32x5 with the real token budget exercises
# the same memory envelope the 256x5 run will see.
set -u
export FASTRL_ROOT=/local_nvme1/mborjigi/fastrl
source "$FASTRL_ROOT/examples/sql/cfg_probe_lib.sh"
OUT=$FASTRL_ROOT/output/train_cfg_sweep
mkdir -p "$OUT"

STEPS=2; MAXSEC=${MAXSEC:-1500}; STALL=${STALL:-240}
COMMON=(NGPUS=4 TP=1 TRAIN_PROMPT_BSZ=32 MINI_BSZ=32 N_RESP=5 MAX_STEPS=$STEPS
        SPECULATIVE=false
        TRAIN_FILE=data/skyrl_sql/bench650.parquet
        VAL_FILE=data/skyrl_sql/validation.parquet)

probe () { local n=$1; shift; probe_run "$n" "$OUT" "$STEPS" "$MAXSEC" "$STALL" -- \
             "${COMMON[@]}" EXPERIMENT_NAME="$n" "$@"; }

# Measured at 32x5 on 4 GPUs:
#   A2 SP2 budget 4096: update_actor 105.1  old_log_prob 40.5  gen 60.2  step 231.3
#   B  SP1 budget 8192: update_actor 127.3  old_log_prob 42.7  gen 40.6  step 232.5
# SP2 wins the training step by 17%, so the scheduling knobs below are probed on
# top of SP2, not SP1. (Step wall is a wash only because gen differed by 20s
# between the two runs, which SP cannot affect -- that is run-to-run noise.)

# --- does a resident optimizer fit if the engine is squeezed? ------------
probe C2_optres_m030  ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.30 OPTIM_OFFLOAD=False
probe C3_optres_m020  ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.20 OPTIM_OFFLOAD=False

# --- gradient-neutral FSDP scheduling knobs on top of A2 ----------------
probe G_reshard_off   ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 RESHARD_AFTER_FWD=False
probe H_prefetch      ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 FWD_PREFETCH=True
probe K_reshard_pref  ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 RESHARD_AFTER_FWD=False FWD_PREFETCH=True
probe I_budget8k      ULYSSES=2 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.6
probe J_nockpt        ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 GRAD_CKPT=False
probe L_sp4           ULYSSES=4 TOKEN_BUDGET=2048 GPU_MEM_UTIL=0.6
echo "### done3"
