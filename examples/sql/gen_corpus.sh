#!/usr/bin/env bash
# Phase 1 of the drafter adaptation: generate the SkyRL-SQL trajectory corpus.
# Generation only, GPUs 0-1, so it coexists with another user on GPUs 2-3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VIRTUAL_ENV="$PWD/.venv"          # fastrl's venv: needs sglang + skyrl_gym
export PATH="$VIRTUAL_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_HOME=${HF_HOME:-/local_nvme0/mborjigi/hf}
PY="$VIRTUAL_ENV/bin/python"

for split in train validation; do
  echo "=== $split ($(date +%H:%M:%S))"
  "$PY" examples/sql/build_eagle_corpus.py \
      --data "data/skyrl_sql/${split}.parquet" \
      --out  "data/eagle_corpus/${split}.parquet" \
      --samples-per-prompt 3 --tp 2 --concurrency 64
done
echo "=== corpus done ($(date +%H:%M:%S))"
