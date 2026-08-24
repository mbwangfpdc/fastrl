#!/usr/bin/env bash
# Focused follow-up to run_train_cfg_sweep.sh.
#   A2: the SP2 baseline that never produced a number (killed by an external
#       SIGTERM the first time, actor death the second) -- needed to judge B.
#   C2/C3: optimizer resident, but with the engine squeezed hard. Every
#       OPTIM_OFFLOAD=False config died at init with the engine at 0.6: a
#       resident fp32 master+m+v is ~21GB/GPU on 4 GPUs and sglang wanted ~24GB.
#       If no-offload only fits with a tiny KV, that trade has to be measured,
#       not assumed.
set -u
ROOT=/local_nvme1/mborjigi/fastrl
OUT=$ROOT/output/train_cfg_sweep
mkdir -p "$OUT"
BSZ=${BSZ:-32}; NRESP=5; STEPS=2; NG=4

reap () {  # give the previous engine time to actually release the cards
  pkill -9 -u "$USER" -f "sglang::" 2>/dev/null
  pkill -9 -u "$USER" -f "ray::WorkerDict" 2>/dev/null
  pkill -9 -u "$USER" -f "main_fastrl" 2>/dev/null
  sleep 45
}

run () {
  local name=$1; shift
  local log="$OUT/$name.log"
  grep -q "global_step:$STEPS" "$log" 2>/dev/null && { echo "== $name: done, skip"; return; }
  reap
  echo "== $name: $*"
  ( cd "$ROOT" && env CUDA_VISIBLE_DEVICES=0,1,2,3 \
      NGPUS=$NG TP=1 TRAIN_PROMPT_BSZ=$BSZ MINI_BSZ=$BSZ N_RESP=$NRESP MAX_STEPS=$STEPS \
      SPECULATIVE=false EXPERIMENT_NAME="$name" \
      TRAIN_FILE=data/skyrl_sql/bench650.parquet VAL_FILE=data/skyrl_sql/validation.parquet \
      "$@" timeout -k 30 1800 bash examples/grpo_sql_baseline.sh > "$log" 2>&1 )
  local ua ol gn st
  ua=$(grep -oE "timing_s/update_actor:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  ol=$(grep -oE "timing_s/old_log_prob:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  gn=$(grep -oE "timing_s/generate_sequences:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  st=$(grep -oE "timing_s/step:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  if [ -z "$ua" ]; then
    local why="failed"
    grep -q "CUDA out of memory" "$log" && why="OOM"
    grep -q "Not enough memory" "$log" && why="sglang-init-OOM"
    grep -q "ActorUnavailableError\|SYSTEM_ERROR" "$log" && why="$why/actor-died"
    echo "   FAILED ($why)"
  else
    printf "   update_actor %-9.1f old_log_prob %-9.1f gen %-9.1f step %.1f\n" "$ua" "$ol" "$gn" "$st"
  fi
}

run A2_sp2_baseline ULYSSES=2 TOKEN_BUDGET=4096 GPU_MEM_UTIL=0.6
run C2_optres_m030  ULYSSES=1 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.30 OPTIM_OFFLOAD=False
run C3_optres_m020  ULYSSES=1 TOKEN_BUDGET=8192 GPU_MEM_UTIL=0.20 OPTIM_OFFLOAD=False
echo "### done2"
