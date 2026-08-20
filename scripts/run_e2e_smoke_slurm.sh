#!/bin/bash
#
# Oscar GPU smoke test: 2 GRPO steps on the SkyRL-SQL workload, speculative
# decoding OFF, tiny batch. Purpose is narrow -- confirm the whole pipeline
# (sglang multi-turn rollout, the SQL interaction, GRPO update, weight sync)
# actually runs end to end on this node/environment before touching anything
# in RESUME_TLT_DRAFTER.md (the drafter-training fixes, the SD multi-turn
# crash). Not a throughput measurement -- see examples/sql/README.md for the
# tuned 256x5 numbers.
#
#   sbatch scripts/run_e2e_smoke_slurm.sh

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-e2e-smoke
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail

REPO=/oscar/scratch/mborjigi/fastrl
cd "$REPO"

echo "node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_JOB_GPUS:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Ray spawns enough worker processes/threads under the Slurm cgroup to hit
# pthread_create EAGAIN at the default soft nproc limit -- reproduced: every
# core worker aborted simultaneously with "pthread_create failed: Resource
# temporarily unavailable". Same fix as granular-cais-rl's sql_slurm.script.
echo "ulimit -u before: $(ulimit -u)"
ulimit -u "$(ulimit -Hu)"
echo "ulimit -u after:  $(ulimit -u)"

# Oscar provides CUDA via Lmod, not /usr/local/cuda (grpo_sql_baseline.sh's
# default) -- sglang JIT-compiles CUDA-graph kernels at rollout start and
# dies with "Could not find nvcc" without this.
module load cuda/12.9.0-cinr
echo "CUDA_HOME=$CUDA_HOME"

# Ray autodetects the node's FULL core count (l40s gpu-he nodes have 128),
# not this job's cgroup share (--cpus-per-task above). Left alone, verl's
# RayWorkerGroup places bundles sized against the inflated count and the
# placement group can never be satisfied within the actual cgroup -- Ray
# starts fine, then the whole run hangs silently forever with no error
# (reproduced: 1h wall clock, zero output past "Started a local Ray
# instance"). Same fix granular-cais-rl's sql_slurm.script already applies.
export RAY_OVERRIDE_RESOURCES="{\"CPU\":$SLURM_CPUS_PER_TASK}"
echo "RAY_OVERRIDE_RESOURCES=$RAY_OVERRIDE_RESOURCES"

# grpo_sql_baseline.sh pins its own VIRTUAL_ENV internally; unset any
# inherited one anyway so nothing upstream of it can leak in.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true

export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}

SPECULATIVE=false \
NGPUS=4 TP=1 ULYSSES=2 \
TRAIN_PROMPT_BSZ=8 MINI_BSZ=8 N_RESP=5 MAX_STEPS=2 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 \
GPU_MEM_UTIL=0.6 \
EXPERIMENT_NAME="e2e-smoke-${SLURM_JOB_ID}" \
bash examples/grpo_sql_baseline.sh

echo "=== done ==="
