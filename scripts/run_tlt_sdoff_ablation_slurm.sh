#!/bin/bash
#
# Speculative-decoding OFF ablation, otherwise byte-identical to
# scripts/run_tlt_norafter_prod_slurm.sh: same base Qwen/Qwen2.5-7B target,
# same production batch (256x5, 10 steps), same GPU_MEM_UTIL=0.5,
# same ENFORCE_EAGER=true, same nccl_timeout fix. Only SPECULATIVE flips to
# false. This isolates the effect of Adaptive Rollout Engine speculative
# decoding on THIS exact workload/model/batch, holding every other variable
# fixed -- the run_tlt_norafter_prod_slurm.sh vs. the SD-off Coder-Instruct
# 35-step reference in RESUME_TLT_DRAFTER.md is NOT a clean ablation (two
# variables change: model AND speculative.enable), so it can't isolate SD's
# own contribution. This script exists to answer that specific question.
#
#   sbatch scripts/run_tlt_sdoff_ablation_slurm.sh                # MAX_STEPS=10
#   sbatch --export=ALL,MAX_STEPS=5 scripts/run_tlt_sdoff_ablation_slurm.sh

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-tlt-sdoff-ablation
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
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

TRAIN_PROMPT_BSZ=256; MINI_BSZ=256; N_RESP=5
MAX_STEPS=${MAX_STEPS:-10}
SAVE_FREQ=-1
RUN_TAG="tlt-sdoff-ablation-${SLURM_JOB_ID}"

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

export PYTHONUNBUFFERED=1

set +e
timeout 21600 env \
MODEL_PATH=Qwen/Qwen2.5-7B \
SPECULATIVE=false \
DRAFTER_TRAINING=false \
DRAFTER_COLLECT_SGL=false \
ENFORCE_EAGER=true \
NGPUS=4 TP=1 \
TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ MINI_BSZ=$MINI_BSZ N_RESP=$N_RESP \
MAX_STEPS=$MAX_STEPS \
ULYSSES=2 TOKEN_BUDGET=4096 FWD_PREFETCH=True \
GPU_MEM_UTIL=$GPU_MEM_UTIL \
EXPERIMENT_NAME="$RUN_TAG" \
bash examples/grpo_sql_baseline.sh trainer.save_freq=$SAVE_FREQ actor_rollout_ref.nccl_timeout=1800
rc=$?
set -e

echo "=== done (rc=$rc) ==="
# Triage:
#   grep -E '^step:' %x-$SLURM_JOB_ID.out
#   grep -o 'gen throughput (token/s): [0-9.]*' %x-$SLURM_JOB_ID.out
exit $rc
