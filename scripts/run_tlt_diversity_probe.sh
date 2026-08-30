#!/bin/bash
#
# Rollout-only diversity/structure probe, turn-mode x SD 2x2, 256x5.
#
# Why this exists. The single-turn 256x5 ablation
# (run_tlt_singleturn_prod_local.sh) came back with SD-OFF already dropping
# ~92% of groups as zero-signal -- i.e. single-turn collapses on its own,
# without any speculative decoding at all, so it sits at a ceiling where an
# additional SD effect cannot be resolved. That rules single-turn out as a
# discriminator and points the question back at multi-turn, where the stark
# gap actually lives (SD off 52/256 and learning, SD on 252/256 and flat).
#
# The multi-turn SD-ON production run had a striking, so-far-unexplained
# property: response length ~2300-2460 tokens with 60-80% of responses hitting
# the 3000-token cap every step, where single-turn SD-off here sits at 392
# tokens mean and a 3% cap rate. Hypothesis: speculation is overshooting the
# per-turn stop tag (`</sql>`). SD emits several tokens per verify step, so a
# stop string landing mid-accepted-block must be detected and truncated
# explicitly; if it isn't, the turn never terminates, the env never gets to
# reply, and the trajectory runs to the cap as one long degenerate turn. That
# would hurt multi-turn specifically -- which is exactly the observed pattern,
# and exactly what single-turn never exercises, since single-turn has no
# per-turn stop tag to overshoot.
#
# This probe tests that directly and cheaply: rollout ONLY (no training, no
# optimizer, FASTRL_ROLLOUT_ONLY=1), one step, both arms, dumping all 1280
# trajectory texts pre-filter. Then
# `scripts/analyze_group_diversity.py` reports turns/trajectory, closed-`</sql>`
# rate, `<solution>` rate and within-group duplicate rate for each arm.
#
# Predictions that would confirm the hypothesis: SD-ON shows markedly fewer
# `<observation>` turns per trajectory and a lower closed-`</sql>` rate than
# SD-OFF, with correspondingly longer responses. If instead both arms show the
# same turn structure and SD-ON simply has near-identical rollouts within each
# group, the cause is sampling-diversity collapse, not stop-tag overshoot.
#
# Run as a 2x2 (turn mode x SD). The single-turn arms are not redundant with
# the single-turn training ablation: that ablation showed WHAT happens (92% of
# groups collapse even with SD off) but not WHY, and mean reward -0.70 points
# at the -1.0 format-error path dominating -- i.e. the base model emitting
# `<sql>...</sql>` and stopping to wait for an observation that single-turn
# mode never sends. The `<solution>` present_frac below settles that directly,
# and if it is near zero the single-turn collapse is a prompt/protocol artifact
# rather than evidence about speculative decoding either way.
#
#   MULTI_TURN=true  SPECULATIVE=false bash scripts/run_tlt_diversity_probe.sh
#   MULTI_TURN=true  SPECULATIVE=true  bash scripts/run_tlt_diversity_probe.sh
#   MULTI_TURN=false SPECULATIVE=false bash scripts/run_tlt_diversity_probe.sh
#   MULTI_TURN=false SPECULATIVE=true  bash scripts/run_tlt_diversity_probe.sh

set -uo pipefail

REPO=/local_nvme1/mborjigi/fastrl
cd "$REPO"

SPECULATIVE=${SPECULATIVE:-true}
MULTI_TURN=${MULTI_TURN:-true}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
TRAIN_PROMPT_BSZ=256; MINI_BSZ=256; N_RESP=5

if [ "$SPECULATIVE" = "true" ]; then ARM=sdon; else ARM=sdoff; fi
if [ "$MULTI_TURN" = "true" ]; then TURN=mt; else TURN=st; fi
RUN_TAG="tlt-probe-${TURN}-${ARM}"
OUT="$REPO/output/$RUN_TAG"
mkdir -p "$OUT/rollouts"

echo "node=$(hostname) arm=$ARM turn=$TURN rollout-only probe"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
export HF_HOME=${HF_HOME:-/local_nvme0/mborjigi/hf}
export RAY_TMPDIR=${RAY_TMPDIR:-/local_nvme1/mborjigi/raytmp}
export TMPDIR="$RAY_TMPDIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export FASTRL_ROLLOUT_ONLY=1
# See run_tlt_singleturn_prod_local.sh: sglang's strict KV-pool idle self-check
# fires on a 2-token discrepancy and kills the scheduler. Set for parity with
# the training arms (a 1-step probe is unlikely to reach it either way).
export SGLANG_TOLERATE_KV_LEAK=1

SPEC_ARGS=()
if [ "$SPECULATIVE" = "true" ]; then
  SPEC_ARGS+=(SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL)
fi

# multi_turn.enable is appended only for the single-turn arms; left alone,
# grpo_sql_baseline.sh's own hardcoded multi_turn.enable=True applies.
MT_ARGS=()
if [ "$MULTI_TURN" != "true" ]; then
  MT_ARGS+=(actor_rollout_ref.rollout.multi_turn.enable=False)
fi
set +e
timeout 10800 env \
MODEL_PATH=Qwen/Qwen2.5-7B \
"${SPEC_ARGS[@]}" \
SPECULATIVE=$SPECULATIVE \
DRAFTER_TRAINING=false \
DRAFTER_COLLECT_SGL=false \
ENFORCE_EAGER=true \
NGPUS=4 TP=1 \
TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ MINI_BSZ=$MINI_BSZ N_RESP=$N_RESP \
MAX_STEPS=1 \
ULYSSES=2 TOKEN_BUDGET=4096 FWD_PREFETCH=True \
GPU_MEM_UTIL=$GPU_MEM_UTIL \
EXPERIMENT_NAME="$RUN_TAG" \
bash examples/grpo_sql_baseline.sh \
  trainer.save_freq=-1 \
  actor_rollout_ref.nccl_timeout=1800 \
  trainer.rollout_data_dir="$OUT/rollouts" \
  "${MT_ARGS[@]}" \
  > "$OUT/run.log" 2>&1
rc=$?
set -e

echo "=== done arm=$ARM turn=$TURN rc=$rc ==="
ls -la "$OUT/rollouts" 2>/dev/null
for f in "$OUT"/rollouts/*.jsonl; do
  [ -f "$f" ] || continue
  echo "--- $f ---"
  "$REPO/.venv/bin/python" "$REPO/scripts/analyze_group_diversity.py" "$f" --n-resp $N_RESP
done
exit $rc
