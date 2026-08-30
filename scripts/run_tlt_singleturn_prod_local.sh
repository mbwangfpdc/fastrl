#!/bin/bash
#
# Single-turn SD on/off ablation at PRODUCTION batch size (256x5), on pistachio.
#
# This is the experiment RESUME_TLT_DRAFTER.md's "UPDATE 2026-08-29" names as
# the next step. The confirmed multi-turn result is a stark gap -- SD ON drops
# 252/256 groups as zero-signal on every step and never learns, SD OFF drops
# ~52/256 and climbs -0.32 -> +0.45. The open question is whether that gap is a
# general property of speculative decoding on this workload or something
# specific to the multi-turn tool-call/observation integration. The 16x4
# single-turn smoke could not answer it (both arms collapsed at ~81-87%, but 16
# groups is far too small a sample to distinguish "no gap" from "gap invisible
# at this scale").
#
# ONE script parameterized by SPECULATIVE, rather than two scripts, so the two
# arms are byte-identical by construction -- the multi-turn pair
# (run_tlt_norafter_prod_slurm.sh vs run_tlt_sdoff_ablation_slurm.sh) had to be
# kept in sync by hand, and the earlier single-turn pair had already drifted
# (only the SD-on one carried the nccl_timeout override).
#
# Every knob below is copied from those two production scripts. The ONLY
# addition is the trailing `actor_rollout_ref.rollout.multi_turn.enable=False`
# (Hydra takes the last value for a repeated key, so it beats
# grpo_sql_baseline.sh's own hardcoded multi_turn.enable=True).
#
#   SPECULATIVE=false bash scripts/run_tlt_singleturn_prod_local.sh
#   SPECULATIVE=true  bash scripts/run_tlt_singleturn_prod_local.sh
#
# MAX_RESPONSE_LENGTH is left at the stock 3000 on purpose, so the zero-signal
# numbers are directly comparable to the multi-turn production runs. The
# earlier 256x5 single-turn attempt that never finished step 1 in 3h was blamed
# on single-turn's long tail, but it predates the `_generate_with_drafter`
# release_memory hang fix (b793853) which hung EVERY single-turn run regardless
# of SD -- so that diagnosis is confounded and worth retesting at stock before
# distorting the workload. If the tail really does dominate, lower it (e.g.
# MAX_RESPONSE_LENGTH=1024) and RERUN BOTH ARMS -- a cap applied to only one
# arm invalidates the ablation.

set -uo pipefail

REPO=/local_nvme1/mborjigi/fastrl
cd "$REPO"

SPECULATIVE=${SPECULATIVE:-true}
MAX_STEPS=${MAX_STEPS:-10}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
TRAIN_PROMPT_BSZ=256; MINI_BSZ=256; N_RESP=5
SAVE_FREQ=-1

if [ "$SPECULATIVE" = "true" ]; then ARM=sdon; else ARM=sdoff; fi
RUN_TAG="tlt-singleturn-prod-${ARM}"
OUT="$REPO/output/$RUN_TAG"
mkdir -p "$OUT"

echo "node=$(hostname) arm=$ARM steps=$MAX_STEPS maxresp=$MAX_RESPONSE_LENGTH"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

# Ray hits pthread_create EAGAIN at the default soft nproc limit; harmless to
# raise here too.
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

# pistachio has the CUDA toolkit at the default prefix (Oscar needs Lmod).
# sglang JIT-compiles EAGLE kernels and dies with "Could not find nvcc"
# without this, even under ENFORCE_EAGER.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"

# Shells here export granular-cais-rl's venv; grpo_sql_baseline.sh must pick up
# fastrl's own. See CLAUDE.md.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
export HF_HOME=${HF_HOME:-/local_nvme0/mborjigi/hf}
export RAY_TMPDIR=${RAY_TMPDIR:-/local_nvme1/mborjigi/raytmp}
export TMPDIR="$RAY_TMPDIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1

# sglang's idle self-check (scheduler.py check_memory) is a strict equality test
# on the KV pool, and on this single-turn path it fires with a TWO-token
# discrepancy (max_total_num_tokens=135945 vs available+evictable=135943),
# killing the scheduler at step 6 of 10 and leaving the run hung with the GPUs
# idle. Patched to be tolerable via this env var -- see the LOCAL PATCH comment
# at that call site. Set for BOTH arms so the ablation stays symmetric; it only
# downgrades an assertion to a warning and does not touch compute. The 2-token
# accounting leak itself is NOT fixed and still wants a root cause; it looks
# single-turn-specific (10-step multi-turn production runs never hit it).
export SGLANG_TOLERATE_KV_LEAK=1

SPEC_ARGS=()
if [ "$SPECULATIVE" = "true" ]; then
  SPEC_ARGS+=(SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL)
fi

set +e
timeout 21600 env \
MODEL_PATH=Qwen/Qwen2.5-7B \
"${SPEC_ARGS[@]}" \
SPECULATIVE=$SPECULATIVE \
DRAFTER_TRAINING=false \
DRAFTER_COLLECT_SGL=false \
ENFORCE_EAGER=true \
NGPUS=4 TP=1 \
TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ MINI_BSZ=$MINI_BSZ N_RESP=$N_RESP \
MAX_STEPS=$MAX_STEPS \
MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH \
ULYSSES=2 TOKEN_BUDGET=4096 FWD_PREFETCH=True \
GPU_MEM_UTIL=$GPU_MEM_UTIL \
EXPERIMENT_NAME="$RUN_TAG" \
bash examples/grpo_sql_baseline.sh \
  trainer.save_freq=$SAVE_FREQ \
  actor_rollout_ref.nccl_timeout=1800 \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  > "$OUT/run.log" 2>&1
rc=$?
set -e

echo "=== done arm=$ARM rc=$rc ==="
# Triage (the number this experiment exists to produce is the first one):
#   grep -o 'train/zero_signal_groups_dropped:[0-9]*' output/$RUN_TAG/run.log
#   grep -o 'critic/score/mean:[-0-9.]*'              output/$RUN_TAG/run.log
#   grep -o 'accept len: [0-9.]*' output/$RUN_TAG/run.log | sort -u | tail
exit $rc
