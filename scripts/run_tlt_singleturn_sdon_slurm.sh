#!/bin/bash
#
# Diagnostic: does SD's zero-signal-group collapse (RESUME_TLT_DRAFTER.md's
# "UPDATE 2026-08-26") reproduce on the SAME SQL task/dataset/model with
# multi-turn DISABLED, or is it specific to the multi-turn tool-calling loop?
# Isolates "multi-turn" as the only variable: same base Qwen2.5-7B target,
# same matched drafter, same SQL dataset/prompts/reward function as
# run_tlt_norafter_prod_slurm.sh, only difference is
# actor_rollout_ref.rollout.multi_turn.enable=False (appended as a trailing
# CLI override -- Hydra takes the LAST value for a repeated key, so this wins
# over grpo_sql_baseline.sh's own hardcoded multi_turn.enable=True). The SQL
# prompt template already permits going straight to <solution> with no
# exploration ("If you find no further exploration is needed... you MUST
# directly provide the final SQL query solution"), and sql_reward.py's
# compute_score scores the decoded response text directly regardless of how
# many turns produced it -- so single-turn scoring is valid with zero data
# changes.
#
# Small/fast on purpose: this is a yes/no diagnostic (does zero-signal collapse
# appear at all), not a training run. 16x4, 2 steps -- matching smoke scale,
# not the 256x5 production batch. First attempt used 256x5/5 steps and hit
# its own 3h internal timeout without finishing step 1's rollout: single-turn
# generation with only a soft </solution> stop-string and no multi-turn
# checkpoint to bound individual trajectories produces a much worse long tail
# than multi-turn at this batch size (most of 1280 rollouts run toward the
# full 3000-token cap rather than a mix of short/long turns). Smoke scale is
# plenty to see zero-signal collapse if it's there.
#
#   sbatch scripts/run_tlt_singleturn_sdon_slurm.sh

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-tlt-singleturn-sdon
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail

REPO=/oscar/scratch/mborjigi/fastrl
cd "$REPO"

echo "node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_JOB_GPUS:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

echo "ulimit -u before: $(ulimit -u)"
ulimit -u "$(ulimit -Hu)"
echo "ulimit -u after:  $(ulimit -u)"

module load cuda/12.9.0-cinr
echo "CUDA_HOME=$CUDA_HOME"

export RAY_OVERRIDE_RESOURCES="{\"CPU\":$SLURM_CPUS_PER_TASK}"

unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}

TRAIN_PROMPT_BSZ=16; MINI_BSZ=16; N_RESP=4
MAX_STEPS=${MAX_STEPS:-2}
SAVE_FREQ=-1
RUN_TAG="tlt-singleturn-sdon-${SLURM_JOB_ID}"

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

export PYTHONUNBUFFERED=1

set +e
timeout 10800 env \
MODEL_PATH=Qwen/Qwen2.5-7B \
SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL \
SPECULATIVE=true \
DRAFTER_TRAINING=false \
DRAFTER_COLLECT_SGL=false \
ENFORCE_EAGER=true \
NGPUS=4 TP=1 \
TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ MINI_BSZ=$MINI_BSZ N_RESP=$N_RESP \
MAX_STEPS=$MAX_STEPS \
ULYSSES=2 TOKEN_BUDGET=4096 FWD_PREFETCH=True \
GPU_MEM_UTIL=$GPU_MEM_UTIL \
EXPERIMENT_NAME="$RUN_TAG" \
bash examples/grpo_sql_baseline.sh trainer.save_freq=$SAVE_FREQ actor_rollout_ref.nccl_timeout=1800 actor_rollout_ref.rollout.multi_turn.enable=False
rc=$?
set -e

echo "=== done (rc=$rc) ==="
# Triage:
#   grep -o 'train/zero_signal_groups_dropped:[0-9]*' %x-$SLURM_JOB_ID.out
#   grep -o 'accept len: [0-9.]*' %x-$SLURM_JOB_ID.out | sort -u
exit $rc
