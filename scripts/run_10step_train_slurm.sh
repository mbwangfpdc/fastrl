#!/bin/bash
#
# Oscar GPU run: 10 GRPO steps on the SkyRL-SQL workload, using the SAME
# ml-sensitive config as granular-cais-rl's sql_baseline.toml and the
# original FastRL-vs-granular head-to-head (tlt_fastrl_baseline_wip.md) --
# examples/grpo_sql_baseline.sh's defaults ARE that config (see its header
# comment for the field-by-field mapping): lr 1e-6, wd 0.01, max_grad_norm
# 0.5, constant schedule, entropy_coef 0.0, batch 256 x 5 generations, temp
# 0.6 / top_p 0.95, max_prompt_length 4096, max_turns 5,
# use_conversation_multi_turn false, speculative decoding OFF (the
# established fair-shot config -- SD is a throughput-only, adaptive-gate
# question, orthogonal to whether training is correct).
#
# Purpose: a cheap first check, before the full 35-step confirmation run,
# that reward climbs with a similar shape to the published SkyRL-SQL curve
# and granular's own training (examples/sql/README.md's "It learns: 10
# steps, no SD, zero errors" table, gathered at batch 64 on the old node --
# this is the first time it's been checked at the real batch 256 on Oscar).
#
#   sbatch scripts/run_10step_train_slurm.sh
#
# Reward curve: grep '^step:' the output log for critic/score/mean per step.

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-sql-10step
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail

REPO=/oscar/scratch/mborjigi/fastrl
cd "$REPO"

echo "node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_JOB_GPUS:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Ray spawns enough worker processes/threads under the Slurm cgroup to hit
# pthread_create EAGAIN at the default soft nproc limit (reproduced on this
# same partition during the e2e smoke test -- job 5106892).
echo "ulimit -u before: $(ulimit -u)"
ulimit -u "$(ulimit -Hu)"
echo "ulimit -u after:  $(ulimit -u)"

# Oscar provides CUDA via Lmod, not /usr/local/cuda (grpo_sql_baseline.sh's
# default) -- sglang JIT-compiles CUDA-graph kernels at rollout start and
# dies with "Could not find nvcc" without this.
module load cuda/12.9.0-cinr
echo "CUDA_HOME=$CUDA_HOME"

# Ray autodetects the node's full core count (128 on gpu-he l40s nodes), not
# this job's cgroup share -- left alone, RayWorkerGroup placement groups are
# sized against the inflated count and the run hangs forever with no error
# (reproduced: job 5106892, 1h wall clock, zero output past Ray startup).
export RAY_OVERRIDE_RESOURCES="{\"CPU\":$SLURM_CPUS_PER_TASK}"
echo "RAY_OVERRIDE_RESOURCES=$RAY_OVERRIDE_RESOURCES"

# grpo_sql_baseline.sh pins its own VIRTUAL_ENV internally; unset any
# inherited one anyway so nothing upstream of it can leak in.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true

export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}

# Everything below is the script's own defaults (== granular's
# sql_baseline.toml / the original head-to-head config) -- only MAX_STEPS
# and the experiment name are overridden.
MAX_STEPS=10 \
EXPERIMENT_NAME="sql-baseline-10step-${SLURM_JOB_ID}" \
bash examples/grpo_sql_baseline.sh

echo "=== done ==="
