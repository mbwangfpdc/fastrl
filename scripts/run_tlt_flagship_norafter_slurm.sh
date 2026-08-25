#!/bin/bash
#
# Variant of scripts/run_tlt_flagship_slurm.sh: Adaptive Rollout Engine only
# (SPECULATIVE=true, matched paper pair) with Adaptive Drafter training OFF.
#
# Separate script (not env-override of run_tlt_flagship_slurm.sh) because that
# script hardcodes DRAFTER_TRAINING=true/DRAFTER_COLLECT_SGL=true inline before
# the `bash examples/grpo_sql_baseline.sh` call, so exporting the env var
# doesn't override it.
#
# Why: job 5197256 (the flagship smoke, both features on) hung completely
# ~8min into step 1, right after the Adaptive Drafter's background-training
# handshake completed (`Worker 0 cleaning up training`), immediately followed by
# a torch_memory_saver print never seen in any prior run of this repo:
#   "Cannot pause allocation that is not active. tag=kv_cache"
# GPU util on the node dropped to 0% across all 4 GPUs and stayed there for
# 12h+ with zero further log output; the node's process/thread table for the
# job's cgroup was fully exhausted (every `srun --overlap` exec failed to fork),
# so this is a real leak, not just slowness -- worse than the handoff's own
# prior report of this same flagship config ("2-step 16x4 smoke timed out at
# 1800s (rc=124) without finishing step 2"), which at least got far enough to
# log repeated `accept len: 1.00` lines before its external timeout fired.
#
# This variant isolates the Adaptive Rollout Engine (the paper's other headline
# feature) from the Adaptive Drafter subsystem implicated in the hang above.
#
# RESULT (job 5242734, 2026-08-25): completed cleanly, rc=0, 8m18s, both steps.
# Accept len ranged 2.05-8.50 across decode batches -- NOT the flat 1.00 seen
# with the old mismatched drafter (Coder-Instruct target + base-Qwen drafter).
# Confirms RESUME_TLT_DRAFTER.md's leading hypothesis: the old "SD 1.8x
# slower" result was an artifact of the wrong pair, not a limitation of the
# Adaptive Rollout Engine itself. See RESUME_TLT_DRAFTER.md for the full
# writeup; whether Adaptive Drafter training can be made to coexist with this
# without hanging is still open (use run_tlt_flagship_slurm.sh to reproduce).

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-tlt-norafter-smoke
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:30:00
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

# smoke scale, matching run_tlt_flagship_slurm.sh's SMOKE=1 branch
TRAIN_PROMPT_BSZ=16; MINI_BSZ=16; N_RESP=4; MAX_STEPS=2; SAVE_FREQ=-1
RUN_TAG="tlt-norafter-smoke-${SLURM_JOB_ID}"

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

# 2700s wall-clock cutoff: the SD-off run finishes full 256x5 steps in ~460s,
# and the prior flagship attempt (both features on) got through enough of
# step 1 to log repeated accept-len readings within its 1800s timeout, so
# 45 min is ample margin for a 16x4 2-step smoke if this is actually alive.
set +e
timeout 2700 env \
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
bash examples/grpo_sql_baseline.sh trainer.save_freq=$SAVE_FREQ
rc=$?
set -e

echo "=== done (rc=$rc) ==="
# Triage (this job's own log is still open for writing, so grep it after the
# fact rather than from inside this script):
#   grep -o 'accept len: [0-9.]*' %x-$SLURM_JOB_ID.out | sort -u | tail
#   grep -E '^step:' %x-$SLURM_JOB_ID.out
exit $rc
