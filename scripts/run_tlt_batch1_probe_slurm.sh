#!/bin/bash
#
# Isolates the residual 0.438 bitwise-match finding in RESUME_TLT_DRAFTER.md's
# `### UPDATE 2026-08-31` from cross-request batch-position effects.
#
# That update fixed the double-prepare_for_decode bug (junk tokens gone,
# lengths match, SD-on-vs-SD-on now 0.992 deterministic) but SD-on-vs-SD-off
# under greedy decoding is still only 0.438 bitwise-identical, not the 1.000
# ceiling correct greedy SD implies. The leading (unproven) explanation there
# is verify-step floating-point numerics: the verify step scores several
# candidate tokens per request in one forward pass, a different kernel shape
# than single-token decode, and near-tie argmaxes can flip. That update also
# notes a CONFOUNDING fact: with SD off entirely, 5 copies of the same prompt
# in one batch already yield only 1.49 distinct greedy outputs -- i.e. this
# engine's greedy output is batch-POSITION-sensitive even with no speculation
# at all. So the 0.438 number conflates two possible effects:
#   (a) cross-request batch-position numerics (proven to exist without SD)
#   (b) verify-step candidate-batching numerics (SD-specific, unproven)
#
# This probe eliminates (a) directly: 4 DISTINCT prompts, 1 response each
# (TRAIN_PROMPT_BSZ=4 N_RESP=1 -- ray_trainer.py's config validation requires
# real_train_batch_size divisible by n_gpus=4, so true train_batch=1 is
# rejected before rollout-only mode ever gets to skip training; 4 prompts is
# the smallest value that clears it). With NGPUS=4 TP=1 the rollout pool is 4
# independent single-GPU sglang engines, and 4 prompts split round-robin means
# each engine ends up handling exactly ONE concurrent request -- so this is
# still true batch=1 at the level that actually does the generating, with no
# duplicate-prompt or cross-request effect possible. If SD-on vs SD-off
# reaches 1.000 here, (a) was the whole story and this SD implementation is
# bitwise-correct at the only scale that matters for its OWN verify-step
# numerics. If it still falls short of 1.000, (b) is real and SD-specific, and
# the "benign floating point" framing in UPDATE 2026-08-31 needs to be
# revisited rather than assumed.
#
# DATA_SHUFFLE=False so both arms deterministically pull the identical first
# prompt from the dataset file, with no dependence on RNG-seed parity between
# the two separate job submissions.
#
# ONE script parameterized by SPECULATIVE (see run_tlt_singleturn_prod_local.sh
# for why: two hand-maintained scripts drift, this doesn't). Single-turn,
# matching the arm the 2026-08-30/31 updates already characterized.
#
# Topology (NGPUS=4 TP=1 ULYSSES=2) is kept IDENTICAL to the already-verified
# single-turn probes rather than shrunk, so this changes exactly one variable
# (batch size) against a known-working baseline.
#
#   SPECULATIVE=false sbatch scripts/run_tlt_batch1_probe_slurm.sh
#   SPECULATIVE=true  sbatch scripts/run_tlt_batch1_probe_slurm.sh
#
# After both land, compare with:
#   .venv/bin/python scripts/compare_runs_bitwise.py \
#       output/tlt-batch1-sdoff/rollouts/*.jsonl output/tlt-batch1-sdon/rollouts/*.jsonl

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-batch1-probe
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
# See run_tlt_singleturn_prod_local.sh: unrelated 2-token KV accounting leak on
# this path; kept for parity even though a 1-request probe is very unlikely to
# reach the step count that trips it.
export SGLANG_TOLERATE_KV_LEAK=1

SPECULATIVE=${SPECULATIVE:-true}
if [ "$SPECULATIVE" = "true" ]; then ARM=sdon; else ARM=sdoff; fi
RUN_TAG="tlt-batch1-${ARM}"
OUT="$REPO/output/$RUN_TAG"
mkdir -p "$OUT/rollouts"

SPEC_ARGS=()
if [ "$SPECULATIVE" = "true" ]; then
  SPEC_ARGS+=(SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL)
fi

set +e
timeout 3000 env \
MODEL_PATH=Qwen/Qwen2.5-7B \
"${SPEC_ARGS[@]}" \
SPECULATIVE=$SPECULATIVE \
DRAFTER_TRAINING=false \
DRAFTER_COLLECT_SGL=false \
ENFORCE_EAGER=true \
NGPUS=4 TP=1 \
TRAIN_PROMPT_BSZ=4 MINI_BSZ=4 N_RESP=1 \
DATA_SHUFFLE=False \
MAX_STEPS=1 \
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

echo "=== done arm=$ARM rc=$rc ==="
ls -la "$OUT/rollouts" 2>/dev/null
exit $rc
