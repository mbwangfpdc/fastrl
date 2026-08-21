#!/bin/bash
#
# Cheap repro of RESUME_TLT_DRAFTER.md's "THE BLOCKER": FastRL's speculative
# decoding crashes on multi-turn whenever speculation actually engages.
# Drafter training OFF (isolates the base SD path from the separate
# opportunistic-drafter-training fixes in commit e945e8f). Tiny batch (8x5)
# so every request starts below bs_threshold and speculation engages
# immediately instead of being gated off the whole run, per the handoff's
# documented repro conditions.
#
#   sbatch scripts/run_sd_crash_repro_slurm.sh
#
# Set ENFORCE_EAGER=True via --export to hit the other documented crash mode
# (KV-buffer-sized-as-whole-pool in triton_backend.forward_decode) instead of
# the CUDA-graph shape mismatch this defaults to:
#   sbatch --export=ALL,ENFORCE_EAGER=True --job-name=fastrl-sd-crash-eager \
#          scripts/run_sd_crash_repro_slurm.sh

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-sd-crash-repro
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL

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
echo "RAY_OVERRIDE_RESOURCES=$RAY_OVERRIDE_RESOURCES"

unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}

# Tiny batch, our own trained (GQA, full-context) drafter, drafter training
# OFF, bs_threshold at its default 32 -- per the handoff, 8x5 means every
# request starts below threshold so speculation engages immediately instead
# of being gated off for the whole run (which is what happens at the real
# 256x5 batch and why this was never caught there).
NGPUS=4 TP=1 ULYSSES=2 TOKEN_BUDGET=2560 \
TRAIN_PROMPT_BSZ=8 MINI_BSZ=8 N_RESP=5 MAX_STEPS=2 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=1024 \
GPU_MEM_UTIL=0.45 \
ENFORCE_EAGER=${ENFORCE_EAGER:-False} \
SPECULATIVE=true SPEC_MODEL_PATH=/oscar/scratch/mborjigi/eagle-drafters/sql-cold-e3 \
DRAFTER_TRAINING=False \
EXPERIMENT_NAME="sd-crash-repro-${SLURM_JOB_ID}" \
bash examples/grpo_sql_baseline.sh

echo "=== done (no crash) ==="
