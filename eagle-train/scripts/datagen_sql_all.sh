#!/usr/bin/env bash
# Phase 2 driver: cache hidden states for both splits on 2 GPUs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for split in train validation; do
  echo "=== $split ($(date +%H:%M:%S))"
  SPLIT=$split NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/datagen_sql.sh 2>&1 \
    | grep -iv "^warning\|FutureWarning\|compilers.C\|it/s\]" | tail -6
done
echo "=== phase 2 done ($(date +%H:%M:%S))"
du -sh /local_nvme1/mborjigi/eagle-cache/* 2>/dev/null
