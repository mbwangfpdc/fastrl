#!/usr/bin/env bash
# Sweep 7 -- can the engine fraction go past 0.7 with SD off?
# Prior 256x5 data suggests the knee is already behind us (gen 269s @0.5 ->
# ~146-208s @0.6-0.7, flat within noise), and training still needs ~42GiB of
# 44.4 after the engine sleeps. 0.8 gives the engine 35.5GB / 354k tokens of KV.
# Two failure modes to watch: sglang refusing at init, or the trainer OOMing
# because the engine did not hand all of it back.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VIRTUAL_ENV="${FASTRL_VENV:-$PWD/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
OUT=output/opt_sweep; mkdir -p "$OUT"
for f in 0.8; do
  name="g4_mem${f/./}_sdoff"
  log="$OUT/${name}.log"
  echo "== $name (gpu_mem_util=$f)"
  timeout -k 30 4200 env SPECULATIVE=false GPU_MEM_UTIL=$f \
      MAX_STEPS=2 EXPERIMENT_NAME="opt-$name" SKYRL_SQL_DUMP_PATH="" \
      bash examples/grpo_sql_baseline.sh >"$log" 2>&1
  rc=$?
  pkill -9 -f main_fastrl >/dev/null 2>&1 || true
  pkill -9 -f "sglang::scheduler" >/dev/null 2>&1 || true
  "$VIRTUAL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
  sleep 20
  echo "== $name rc=$rc steps: $(grep -o 'timing_s/step:[0-9.]*' "$log" | tr '\n' ' ')"
  echo "   gen: $(grep -o 'timing_s/generate_sequences:[0-9.]*' "$log" | tr '\n' ' ')"
  grep -q "OutOfMemoryError\|Not enough memory" "$log" && echo "   -> hit a memory limit"
done
