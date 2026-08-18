#!/usr/bin/env bash
# Phase 3 + ablation. The warm-started run began at 7.5% draft accuracy, which is
# roughly cold-start territory -- mit-han-lab's drafter was trained on Qwen2.5-7B
# *base* hidden states while ours come from Coder-7B-Instruct, so the transfer may
# be worth little. This runs both arms so the question is answered by measurement
# rather than assumed either way.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for arm in warm cold; do
  echo "=== $arm ($(date +%H:%M:%S))"
  if [ "$arm" = "warm" ]; then W="$PWD/models/drafter-mit-han-lab/pytorch_model.bin"; else W=""; fi
  WARM_START="$W" EPOCHS=3 \
    CKPT_PATH=/local_nvme1/mborjigi/eagle-ckpt/sql-coder7b-$arm \
    bash scripts/train_sql.sh 2>&1 | tr '\r' '\n' \
    | grep -Ei "warm start|Epoch [0-9]+/3: 100%|OutOfMemory|Traceback" | tail -6
done
echo "=== phase 3 done ($(date +%H:%M:%S))"
