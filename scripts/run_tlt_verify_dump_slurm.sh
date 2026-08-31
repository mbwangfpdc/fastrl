#!/bin/bash
#
# Next step after RESUME_TLT_DRAFTER.md's `### UPDATE 2026-08-31b`: is
# `verify_tree_greedy`'s own accept decision wrong given its own inputs (a
# kernel-logic bug), or is the input it's handed (`target_predict`, the
# target model's argmax at each candidate position) already different from
# what a plain non-speculative decode would compute at the identical context
# (context corruption or genuine floating-point drift)?
#
# Dumps every greedy verify round's kernel inputs AND outputs to
# FASTRL_DEBUG_VERIFY_DIR (see eagle_info.py's EagleVerifyInput.verify, greedy
# branch) for one short SD-on rollout-only generation, then checks each round
# with scripts/check_verify_tree_greedy.py, which independently re-walks the
# same tensors in pure Python (verified against the kernel's own unit test in
# that script's header) and diffs the result against what the kernel actually
# returned.
#
# SD-off is not needed here -- it never calls verify_tree_greedy at all, so
# there is nothing to dump on that arm.
#
# Same topology and batch=4/n_resp=1 isolation as run_tlt_batch1_probe_slurm.sh
# (see that script's header for why: true single-request-per-engine batch=1,
# despite ray_trainer.py's config validation rejecting train_batch_size=1
# outright). MAX_RESPONSE_LENGTH capped low purely to keep the dump (and the
# job) small and fast -- this test is about whether the kernel agrees with
# itself, which needs only a handful of verify rounds, not a full generation.
#
#   sbatch scripts/run_tlt_verify_dump_slurm.sh

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-verify-dump
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -uo pipefail

REPO=/oscar/scratch/mborjigi/fastrl
cd "$REPO"

echo "node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_JOB_GPUS:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
module load cuda/12.9.0-cinr
echo "CUDA_HOME=$CUDA_HOME"

export RAY_OVERRIDE_RESOURCES="{\"CPU\":$SLURM_CPUS_PER_TASK}"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}
export PYTHONUNBUFFERED=1
export FASTRL_ROLLOUT_ONLY=1
export SGLANG_TOLERATE_KV_LEAK=1

RUN_TAG="tlt-verify-dump"
OUT="$REPO/output/$RUN_TAG"
mkdir -p "$OUT/rollouts"
export FASTRL_DEBUG_VERIFY_DIR="$OUT/verify_rounds"
rm -rf "$FASTRL_DEBUG_VERIFY_DIR"
mkdir -p "$FASTRL_DEBUG_VERIFY_DIR"

set +e
timeout 3000 env \
MODEL_PATH=Qwen/Qwen2.5-7B \
SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL \
SPECULATIVE=true \
DRAFTER_TRAINING=false \
DRAFTER_COLLECT_SGL=false \
ENFORCE_EAGER=true \
NGPUS=4 TP=1 \
TRAIN_PROMPT_BSZ=4 MINI_BSZ=4 N_RESP=1 \
DATA_SHUFFLE=False \
MAX_STEPS=1 \
MAX_RESPONSE_LENGTH=128 \
ULYSSES=2 TOKEN_BUDGET=4096 FWD_PREFETCH=True \
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5} \
EXPERIMENT_NAME="$RUN_TAG" \
bash examples/grpo_sql_baseline.sh \
  trainer.save_freq=-1 \
  actor_rollout_ref.nccl_timeout=1800 \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  actor_rollout_ref.rollout.temperature=0 \
  trainer.rollout_data_dir="$OUT/rollouts" \
  > "$OUT/run.log" 2>&1
rc=$?
set -e

echo "=== done rc=$rc ==="
ls "$FASTRL_DEBUG_VERIFY_DIR" | wc -l
echo "verify rounds dumped above; checking kernel self-consistency:"
.venv/bin/python scripts/check_verify_tree_greedy.py "$FASTRL_DEBUG_VERIFY_DIR"
exit $rc
