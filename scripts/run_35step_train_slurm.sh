#!/bin/bash
#
# Oscar GPU run: the full 35-step GRPO confirmation run on the SkyRL-SQL
# workload, same ml-sensitive config as granular-cais-rl's sql_baseline.toml
# (see scripts/run_10step_train_slurm.sh's header for the field-by-field
# mapping -- this is that same script with MAX_STEPS bumped to the real
# value and periodic checkpointing turned on).
#
# Run scripts/run_10step_train_slurm.sh FIRST and confirm the reward curve
# looks right before spending ~14-20 GPU-hours on this one.
#
#   sbatch scripts/run_35step_train_slurm.sh
#
# Reward curve: grep '^step:' the output log for critic/score/mean per step.

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-sql-35step
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
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

# Everything below except MAX_STEPS/EXPERIMENT_NAME is the script's own
# defaults (== granular's sql_baseline.toml / the original head-to-head
# config). trainer.save_freq is a positional Hydra override (grpo_sql_
# baseline.sh hardcodes -1, i.e. no checkpoints) -- turned on here because
# this run is long enough (~14-20h) that losing all progress to a preemption
# or a node issue partway through would be expensive; not part of the
# ml-sensitive config being matched.
MAX_STEPS=35 \
EXPERIMENT_NAME="sql-baseline-35step-${SLURM_JOB_ID}" \
bash examples/grpo_sql_baseline.sh trainer.save_freq=10

echo "=== done ==="
