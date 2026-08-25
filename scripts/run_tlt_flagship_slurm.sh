#!/bin/bash
#
# Oscar GPU run: TLT's flagship configuration on the SkyRL-SQL workload --
# BOTH headline components of the ASPLOS'26 paper (arXiv:2511.16665) enabled,
# CUDA graphs deliberately OFF to match granular-cais-rl's vllm_enforce_eager.
#
#   Adaptive Drafter (paper 4)      -> DRAFTER_TRAINING + DRAFTER_COLLECT_SGL
#   Adaptive Rollout Engine (5)     -> SPECULATIVE=true; BEG-MAB tuner and the
#                                       elastic bs_threshold=32 gate are already
#                                       the fastrl defaults, no flag needed
#   Bucketed CUDAGraph Capture (5.1)-> N/A by design: ENFORCE_EAGER=true
#
# Model pair is the PAPER'S OWN (examples/grpo_7B.sh), not our SQL default:
#   target  Qwen/Qwen2.5-7B                    (base, as in the paper)
#   drafter mit-han-lab/Qwen2.5-7B-Eagle-RL    (their published EAGLE drafter)
# grpo_sql_baseline.sh otherwise defaults MODEL_PATH to Qwen2.5-Coder-7B-Instruct,
# which does NOT match that drafter -- see RESUME_TLT_DRAFTER.md "the mismatch".
#
#   sbatch scripts/run_tlt_flagship_slurm.sh
#
# READ RESUME_TLT_DRAFTER.md "STATUS 2026-08-25" FIRST. As of that date this
# config runs without crashing but speculation delivers NOTHING (accept len
# 1.00), so this job is a DIAGNOSTIC, not a known-good production run. Start
# with the smoke variant below before spending real GPU-hours.

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-tlt-flagship
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

# Ray hits pthread_create EAGAIN at the default soft nproc limit under the
# Slurm cgroup (job 5106892). Same rationale as run_35step_train_slurm.sh.
echo "ulimit -u before: $(ulimit -u)"
ulimit -u "$(ulimit -Hu)"
echo "ulimit -u after:  $(ulimit -u)"

# Oscar provides CUDA via Lmod, not /usr/local/cuda. sglang JIT-compiles
# kernels for the EAGLE path and dies with "Could not find nvcc" without this.
# Still required with ENFORCE_EAGER: the drafter path compiles kernels beyond
# the CUDA-graph capture that enforce_eager disables.
module load cuda/12.9.0-cinr
echo "CUDA_HOME=$CUDA_HOME"

# Ray autodetects the node's full core count, not this job's cgroup share;
# left alone the placement groups are oversized and the run hangs silently.
export RAY_OVERRIDE_RESOURCES="{\"CPU\":$SLURM_CPUS_PER_TASK}"

unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}

# --- scale -------------------------------------------------------------
# SMOKE=1 runs a 2-step 16x4 sanity check (~30 min) instead of the real thing.
# Do this first: on pistachio the full-feature config timed out at exactly
# this scale, so proving it completes 2 steps is the gate for the big run.
if [ "${SMOKE:-0}" = "1" ]; then
  TRAIN_PROMPT_BSZ=16; MINI_BSZ=16; N_RESP=4; MAX_STEPS=2; SAVE_FREQ=-1
  RUN_TAG="tlt-flagship-smoke-${SLURM_JOB_ID}"
else
  TRAIN_PROMPT_BSZ=256; MINI_BSZ=256; N_RESP=5; MAX_STEPS=35; SAVE_FREQ=35
  RUN_TAG="tlt-flagship-35step-${SLURM_JOB_ID}"
fi

# --- memory ------------------------------------------------------------
# 0.5, not the 0.6 the SD-off tuning settled on: grpo_sql_baseline.sh's own
# comment requires 0.5 when SD runs with an MHA drafter, and the Spot Trainer
# adds ~7 GiB/GPU on top (measured: actor 15.03 GiB with vs ~7.9 without).
# That combination OOMed a 44 GiB L40S at 0.5 + 16x5 once already -- if this
# OOMs, drop to 0.45 and/or lower N_RESP before touching anything else.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

# Tuned execution knobs carried over from the SD-off 35-step run (they do not
# change the gradient): SP2 + forward_prefetch was ~18% faster on update_actor.
# See examples/sql/TRAIN_CONFIG_TUNING.md.
MODEL_PATH=Qwen/Qwen2.5-7B \
SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL \
SPECULATIVE=true \
DRAFTER_TRAINING=true \
DRAFTER_COLLECT_SGL=true \
DRAFTER_INTERVAL=${DRAFTER_INTERVAL:-10} \
ENFORCE_EAGER=true \
NGPUS=4 TP=1 \
TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ MINI_BSZ=$MINI_BSZ N_RESP=$N_RESP \
MAX_STEPS=$MAX_STEPS \
ULYSSES=2 TOKEN_BUDGET=4096 FWD_PREFETCH=True \
GPU_MEM_UTIL=$GPU_MEM_UTIL \
EXPERIMENT_NAME="$RUN_TAG" \
bash examples/grpo_sql_baseline.sh trainer.save_freq=$SAVE_FREQ

echo "=== done ==="
# Triage, in priority order:
#   grep -o 'accept len: [0-9.]*'   %x-$SLURM_JOB_ID.out | sort -u | tail
#       -> 1.00 everywhere means speculation is still contributing nothing;
#          that is the open bug, do not read a speed number as meaningful.
#   grep -E 'Starting training: DP ranks|activate_training_model' %x-*.out
#       -> Spot Trainer actually engaging (Adaptive Drafter alive).
#   grep -E '^step:' %x-$SLURM_JOB_ID.out
#       -> reward curve; critic/score/mean should climb off its step-1 value.
