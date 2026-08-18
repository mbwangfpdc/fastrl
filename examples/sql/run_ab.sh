#!/usr/bin/env bash
# A/B the FastRL adaptive-SD speedup on the SkyRL-SQL workload.
# Runs the same config twice, speculative decoding off then on, and dumps
# scored rollouts for inspection.
#
#   bash examples/sql/run_ab.sh            # defaults below
#   STEPS=10 BSZ=128 bash examples/sql/run_ab.sh
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

STEPS=${STEPS:-6}
BSZ=${BSZ:-64}

for SD in ${ORDER:-false true}; do
  echo "############ SPECULATIVE=$SD start $(date +%H:%M:%S) ############"
  SKYRL_SQL_DUMP_PATH="$PWD/output/rollouts_sd${SD}.jsonl" \
  SKYRL_SQL_DUMP_N=400 \
  TRAIN_PROMPT_BSZ="$BSZ" MINI_BSZ="$BSZ" MAX_STEPS="$STEPS" \
  SPECULATIVE="$SD" EXPERIMENT_NAME="ab-sd${SD}" \
  bash examples/grpo_sql_baseline.sh > "output/ab_sd${SD}.log" 2>&1
  echo "############ SPECULATIVE=$SD exit=$? $(date +%H:%M:%S) ############"
done
