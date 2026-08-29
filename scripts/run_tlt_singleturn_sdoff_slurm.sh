#!/bin/bash
#
# Single-turn SD-OFF counterpart to run_tlt_singleturn_sdon_slurm.sh -- see
# that script's header for the full rationale. Byte-identical except
# SPECULATIVE=false (no SPEC_MODEL_PATH needed) and no nccl_timeout override
# (only relevant when the earlier DP-straggler issue was in play; harmless
# either way but omitted for a minimal diff against
# run_tlt_sdoff_ablation_slurm.sh's multi-turn baseline).
#
#   sbatch scripts/run_tlt_singleturn_sdoff_slurm.sh

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-tlt-singleturn-sdoff
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
RUN_TAG="tlt-singleturn-sdoff-${SLURM_JOB_ID}"

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

export PYTHONUNBUFFERED=1

set +e
timeout 10800 env \
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
bash examples/grpo_sql_baseline.sh trainer.save_freq=$SAVE_FREQ actor_rollout_ref.rollout.multi_turn.enable=False
rc=$?
set -e

echo "=== done (rc=$rc) ==="
exit $rc
