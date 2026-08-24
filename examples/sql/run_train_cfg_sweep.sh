#!/usr/bin/env bash
# Find an efficient TRAINING config for the 35-step SkyRL-SQL e2e run.
#
# Config-only (no FastRL implementation changes). Small batch, 2 steps each, so
# step 2 is a warm measurement. We care about update_actor + old_log_prob; gen
# is held constant by fixing the engine knobs.
#
# The hypotheses, in order of expected payoff:
#   ulysses SP=2 costs an all-to-all per layer and halves per-GPU token count;
#   on 4 GPUs with dp=4 it may be pure overhead for a 7k-token sequence.
#   param/optim offload move GB over PCIe every step -- a sibling investigation
#   found PyTorch's FSDP2 CPU-offload path is memcpy-bound (88% self-CPU in
#   cudaMemcpyAsync), so turning it off should be a large win IF it fits.
set -u
ROOT=/local_nvme1/mborjigi/fastrl
OUT=$ROOT/output/train_cfg_sweep
mkdir -p "$OUT"

BSZ=${BSZ:-32}; NRESP=${NRESP:-5}; STEPS=${STEPS:-2}; NG=${NG:-4}

run () {   # name, then KEY=VAL env overrides
  local name=$1; shift
  local log="$OUT/$name.log"
  if [ -f "$log" ] && grep -q "global_step:$STEPS" "$log" 2>/dev/null; then
    echo "== $name: already done, skipping"; return
  fi
  echo "== $name: $*"
  ( cd "$ROOT" && env CUDA_VISIBLE_DEVICES=0,1,2,3 \
      NGPUS=$NG TP=1 TRAIN_PROMPT_BSZ=$BSZ MINI_BSZ=$BSZ N_RESP=$NRESP MAX_STEPS=$STEPS \
      SPECULATIVE=false EXPERIMENT_NAME="$name" \
      TRAIN_FILE=data/skyrl_sql/bench650.parquet VAL_FILE=data/skyrl_sql/validation.parquet \
      "$@" timeout -k 30 2400 bash examples/grpo_sql_baseline.sh > "$log" 2>&1 )
  local rc=$?
  local ua ol gn
  ua=$(grep -oE "timing_s/update_actor:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  ol=$(grep -oE "timing_s/old_log_prob:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  gn=$(grep -oE "timing_s/generate_sequences:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  local st=$(grep -oE "timing_s/step:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  if [ -z "$ua" ]; then
    local why="rc=$rc"
    grep -q "CUDA out of memory" "$log" && why="OOM"
    grep -q "Not enough memory" "$log" && why="sglang-init-OOM"
    echo "   FAILED ($why)"
  else
    printf "   update_actor %-8s old_log_prob %-8s gen %-8s step %s\n" "$ua" "$ol" "$gn" "$st"
  fi
  # reap engine strays between configs so the next one starts clean
  pkill -9 -u "$USER" -f "sglang::" 2>/dev/null; pkill -9 -u "$USER" -f "ray::WorkerDict" 2>/dev/null
  sleep 20
}

echo "### baseline (current defaults): SP2, budget 4096, offloads ON"
run A_sp2_off_on   ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6

echo "### SP1 (budget must cover the whole 7096-token sequence)"
run B_sp1_off_on   ULYSSES=1 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.6

echo "### SP1, optimizer resident (params still offloaded)"
run C_sp1_optres   ULYSSES=1 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.6 OPTIM_OFFLOAD=False

echo "### SP1, nothing offloaded -- needs headroom at sglang init"
run D_sp1_nooff    ULYSSES=1 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.85 PARAM_OFFLOAD=False OPTIM_OFFLOAD=False

echo "### SP2 with optimizer resident, as a cross-check"
run E_sp2_optres   ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6 OPTIM_OFFLOAD=False

echo "### SP1, larger microbatch budget (fewer, bigger microbatches)"
run F_sp1_bigbudget ULYSSES=1 TOKEN_BUDGET=16384 GPU_MEM_UTIL=0.6 OPTIM_OFFLOAD=False

echo "### done"
