#!/usr/bin/env bash
# Phase 2 of the drafter adaptation: cache target hidden states for the
# SkyRL-SQL corpus built in Phase 1.
#
# Differences from upstream scripts/datagen_eagle2.sh:
#   * no Slurm -- upstream derives MASTER_ADDR from scontrol and assumes
#     --nproc_per_node=8. This is single-node torchrun on NGPUS (default 2), so
#     it coexists with another user on the remaining cards.
#   * max_length 8192, not the 2048 in the upstream example: that is our actual
#     rollout window, and truncating there would throw away most of the
#     multi-turn trajectories (train mean is ~2271 tokens, tail to 8k).
#   * sample_ratio 1.0 -- the corpus is already exactly the data we want, so
#     there is nothing to subsample.
#   * base_model_path must be a DIRECTORY: the trainer opens
#     model.safetensors.index.json by path, so a bare HF repo id fails.
#
# Rank sharding is `for i in range(rank, n, world_size)` (eagle_datagen.py:330),
# so each rank writes its own disjoint shard into save_dir.
#
# Expected output ~32GB (train) / ~30GB (validation) of last-layer hidden states
# at 3584 dims, bf16.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# eagle-train's OWN venv (torch 2.6 / transformers 4.51), never an inherited one.
export VIRTUAL_ENV="${EAGLE_VENV:-$ROOT/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
PY="$VIRTUAL_ENV/bin/python"

export HF_HOME=${HF_HOME:-/local_nvme0/mborjigi/hf}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=${NGPUS:-2}

SPLIT=${SPLIT:-train}
# The directory name MUST contain "qwen". eagle_datagen.py picks chat separators
# by substring-matching the *path* (not the config), and its else-branch leaves
# model_type=None, which later dies with "Unsupported model type: None". A path
# like ".../target-coder7b-instruct" looks fine and fails.
BASE_MODEL_PATH=${BASE_MODEL_PATH:-$ROOT/models/qwen2.5-coder-7b-instruct-target}
DATA_PATH=${DATA_PATH:-$ROOT/../data/eagle_hf/${SPLIT}}
# The `data-8K` subdirectory is load-bearing, not cosmetic. eagle_trainer.py
# infers the padding length by listing SUBDIRECTORIES of --data_path, splitting
# the name on "-", and parsing field [1] as e.g. "8K" -> 8192. With .pt files
# sitting directly in --data_path there are no subdirectories, so it silently
# keeps its 2048 default and the collator then computes a negative pad width:
#   RuntimeError: Trying to create tensor with negative dimension -778
SAVE_DIR=${SAVE_DIR:-/local_nvme1/mborjigi/eagle-cache/sql-coder7b-${SPLIT}/data-8K}
MAX_LENGTH=${MAX_LENGTH:-8192}
SAMPLE_RATIO=${SAMPLE_RATIO:-1.0}

mkdir -p "$SAVE_DIR"
echo "==> interpreter : $PY"
echo "==> split       : $SPLIT"
echo "==> data        : $DATA_PATH"
echo "==> save_dir    : $SAVE_DIR"
echo "==> gpus        : $CUDA_VISIBLE_DEVICES (nproc=$NGPUS)"

"$VIRTUAL_ENV/bin/torchrun" --nnodes=1 --nproc_per_node="$NGPUS" \
    --master_port="${MASTER_PORT:-12355}" \
    eagle_datagen.py \
    model.base_model_path="$BASE_MODEL_PATH" \
    data.data_path="$DATA_PATH" \
    data.save_dir="$SAVE_DIR" \
    data.max_length="$MAX_LENGTH" \
    data.sample_ratio="$SAMPLE_RATIO" \
    data.mode=eagle2
