#!/usr/bin/env bash
# Phase 3: warm-started drafter training on the cached SkyRL-SQL hidden states.
#
# Differences from upstream scripts/train_eagle2.sh:
#   * --warm_start_ckpt: initialise from mit-han-lab/Qwen2.5-7B-Eagle-RL rather
#     than from scratch. Verified architecture-identical to the config derived
#     from our target (0 shape mismatches, only lm_head missing, which
#     _load_base_model_weights supplies). This is adaptation, not a cold start.
#   * 3 epochs, not 20. Upstream deliberately sets a large epoch count and
#     expects you to stop at convergence; from a warm start we need far less.
#   * config/deepspeed_config_sql.json, not the stock config -- see the comment
#     block in that file. The stock warmup (12000 steps) exceeds our entire run.
#   * 2 GPUs, no Slurm.
#
# The trainer logs only value/prob losses -- there is NO acceptance metric here,
# so a falling loss is necessary but not sufficient. Checkpoint selection happens
# in Phase 4 via scripts/bench_speculative_decoding.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VIRTUAL_ENV="${EAGLE_VENV:-$ROOT/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export HF_HOME=${HF_HOME:-/local_nvme0/mborjigi/hf}
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# No wandb account is wired up on this box; offline keeps the trainer's logger
# from blocking on a login prompt mid-run.
export WANDB_MODE=${WANDB_MODE:-offline}

NGPUS=${NGPUS:-2}
EPOCHS=${EPOCHS:-3}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-$ROOT/models/qwen2.5-coder-7b-instruct-target}
WARM_START=${WARM_START:-$ROOT/models/drafter-mit-han-lab/pytorch_model.bin}
DATA_PATH=${DATA_PATH:-/local_nvme1/mborjigi/eagle-cache/sql-coder7b-train}
CKPT_PATH=${CKPT_PATH:-/local_nvme1/mborjigi/eagle-ckpt/sql-coder7b-warm}

mkdir -p "$CKPT_PATH"
echo "==> interpreter : $VIRTUAL_ENV/bin/python"
echo "==> warm start  : ${WARM_START:-<none, cold start>}"
echo "==> data        : $DATA_PATH"
echo "==> ckpt        : $CKPT_PATH"

WARM_ARGS=()
[[ -n "$WARM_START" ]] && WARM_ARGS=(--warm_start_ckpt "$WARM_START")

"$VIRTUAL_ENV/bin/deepspeed" --num_gpus="$NGPUS" eagle_trainer.py \
    --deepspeed_config config/deepspeed_config_sql.json \
    --base_model_path "$BASE_MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$CKPT_PATH" \
    --project_name FastRL-SQL-Drafter \
    --experiment_name eagle2-warm-coder7b \
    --batch_size 1 \
    --epochs "$EPOCHS" \
    --precision bf16 \
    "${WARM_ARGS[@]}"
