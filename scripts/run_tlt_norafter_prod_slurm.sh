#!/bin/bash
#
# Production-scale run of the paper's matched pair (target Qwen/Qwen2.5-7B,
# drafter mit-han-lab/Qwen2.5-7B-Eagle-RL) with the Adaptive Rollout Engine
# on (SPECULATIVE=true) and Adaptive Drafter training OFF -- the confirmed
# side of run_tlt_flagship_norafter_slurm.sh's 16x4 smoke result (job 5242734:
# accept len 2.05-8.50, clean completion), now at the real 256x5 batch used by
# the SD-off 35-step reference run, to measure reward, generation length, and
# latency against that reference over MAX_STEPS steps (default 10).
#
# Drafter training stays off deliberately: run_tlt_flagship_slurm.sh (both
# features on) hangs the whole node at smoke scale already (job 5197256, see
# RESUME_TLT_DRAFTER.md's "UPDATE 2026-08-25 (Oscar)") -- that bug is being
# tackled separately, after this component is understood on its own.
#
# Comparison points already on record (RESUME_TLT_DRAFTER.md):
#   SD-off, Coder-Instruct target, 256x5, 35 steps: reward -0.168 -> 0.562,
#   step time 1150s -> ~460s as responses shorten (mean resp length 755->471).
#
#   sbatch scripts/run_tlt_norafter_prod_slurm.sh                # MAX_STEPS=10
#   sbatch --export=ALL,MAX_STEPS=5 scripts/run_tlt_norafter_prod_slurm.sh
#
# Attempt 1 (job 5244584, GPU_MEM_UTIL=0.5, no PYTORCH_CUDA_ALLOC_CONF): got
# through 8/10 steps (2h33m in, ~1080-1220s/step), then a worker OOM'd inside
# the EAGLE draft's KV-index allocation (triton_backend.py init_forward_metadata,
# torch.empty for kv_indices) -- "9.12 GiB reserved by PyTorch but unallocated"
# is the classic fragmentation signature growing across many steps, not a hard
# cap violation. It also lost ALL of steps 1-8's reward/gen-length/timing
# metrics: they only ever landed in the driver process's block-buffered
# stdout, which the OOM's abnormal exit never flushed to disk (tqdm's
# `Training Progress: n/N` survived because tqdm writes to stderr,
# line-buffered by default) -- PYTHONUNBUFFERED=1 below fixes this for future
# runs, matching the same fix already applied for this reason in
# granular-cais-rl's gantt_profile_slurm.script (commit 3e83cf7).
#
# Attempt 2 (job 5251663, added PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# per the OOM message's own suggestion): failed immediately at startup --
# `RuntimeError: TorchMemorySaver is disabled for the current process because
# expandable_segments is not supported yet`. sglang's KV-cache pause/resume
# depends on torch_memory_saver, which explicitly refuses to coexist with
# expandable_segments. Reverted.
#
# Attempt 3 (job 5274983, GPU_MEM_UTIL dropped to 0.40 instead): completed
# step 1 (41m39s -- itself ~2.4x job 5244584's step-1 time, the lower memory
# budget serializing more of the batch), then hung at step 2's
# `dist.barrier()` in `_req_level_generate_sequences`
# (sglang_rollout.py, then line 1431) and died to gloo's 600s default
# timeout. Root cause found and fixed in-tree: that barrier call was unscoped
# (default WORLD group) while the `broadcast_pyobj` two lines below it, and
# every other barrier in that file, explicitly scope to the TP group. At
# TP=1 this made a should-be-trivial per-shard barrier into a full 4-way
# DP-rank rendezvous -- harmless when DP ranks finish close together, fatal
# once one straggles past 10 minutes, which lower GPU_MEM_UTIL made more
# likely by slowing decode. Confirmed original upstream code (`be268f0`, the
# initial commit), not something either Oscar attempt introduced. Fixed by
# scoping it to the TP group like its siblings; see the comment at that call
# site for detail. GPU_MEM_UTIL restored to 0.5 (the value job 5244584 used
# to get 8 clean steps, and the tuned value examples/grpo_sql_baseline.sh's
# own comment says is required with SD + an MHA drafter) since the real fix
# was the barrier, not the memory cut -- 0.40 only masked the symptom while
# generally slowing the run, which would confound the latency numbers this
# probe exists to measure.
#
# Attempt 4 (job 5296652, barrier fix + GPU_MEM_UTIL=0.5): got through 4/10
# steps then died to a DIFFERENT 600s collective timeout -- this one a real
# NCCL ALLREDUCE inside actor training (not the rollout-side gloo barrier from
# attempt 3), watchdog-aborted with SIGABRT. Root cause this time is not a
# bug: DP rank 1's own log showed it still legitimately mid-rollout (Decode
# batch running-req counting down 39 -> 12, still generating) while the other
# 3 ranks had already finished and were waiting at the post-rollout actor
# sync. Multi-turn SQL rollout duration varies a lot per DP shard (different
# prompts draw different turn counts / response lengths), and at production
# batch size (256x5) that skew can genuinely exceed 10 minutes -- this isn't
# something a code fix resolves, the workload just needs a longer collective
# timeout than PyTorch's 600s default. `trainer.nccl_timeout` is a documented
# Hydra key (verl/trainer/config/fastrl_trainer.yaml:184, "you can set it to
# a larger value if you have long-running operations" per its megatron
# sibling's comment) -- passed below via the same CLI-override mechanism
# already used for trainer.save_freq, no code change needed. Needs a `+`
# prefix (`+trainer.nccl_timeout=...`), not plain `trainer.nccl_timeout=...`:
# the trainer's structured config dataclass doesn't declare this field even
# though the yaml sets a default for it, so Hydra's struct mode rejects a
# plain override with "Key 'nccl_timeout' is not in struct" (hit once, in a
# job that failed at Hydra parse time before touching any GPU resource).
#
# Note: no run at production batch size has yet produced the per-step
# `critic/score/mean` / `response_length/mean` console metric lines that job
# 5242734's clean 16x4 smoke run showed -- confirmed this happens even for
# individual steps that completed with no crash at all (steps 3/4 of job
# 5307396 advanced normally per tqdm, zero metric-line matches for either).
# Reconstructible via tqdm's own per-iteration timestamps regardless. Still
# unresolved and not investigated further; not blocking.
#
# Attempt 5 (job 5305493, +trainer.nccl_timeout=1800 syntax fix pending):
# failed at Hydra parse before touching any GPU -- plain `trainer.nccl_timeout=`
# needs a `+` prefix since the field isn't declared in the trainer's structured
# config dataclass. Fixed immediately, no GPU time lost.
#
# Attempt 6 (job 5305914, `+trainer.nccl_timeout=1800`, parses fine): OOM'd
# immediately at step 0 (before completing anything), same fragmentation
# signature as attempt 1 but at the very start rather than after several
# steps -- confirms the OOM isn't step-count-dependent, just allocator-layout
# luck (later root-caused properly, see below).
#
# Attempt 7 (job 5307396, retry, same config): got through 6/10 steps this
# time before the same OOM. Metrics for steps 1, 2, 5 landed
# (critic/score/mean pinned at -0.80 across all three, response_length/mean
# ~2280-2430 with 65-80% of responses clip_ratio'd at the 3000-token cap,
# zero_signal_groups_dropped 252/256 every step) -- the base (non-instruct)
# Qwen2.5-7B cold-starting on multi-turn SQL is a real, striking quality
# finding independent of the OOM investigation: it isn't reliably terminating
# with <solution> and produces near-zero learning signal in early steps.
#
# ROOT CAUSE FOUND for the OOM (superseding the "fragmentation" framing of
# attempts 1/3/6/7 above): AsyncEngine(...) in sglang_rollout.py's
# _init_inference_engine never passed context_length, so sglang's ModelConfig
# fell back to the model's own native max_position_embeddings (131072 for
# Qwen2.5-7B, confirmed via both the target's and drafter's config.json) --
# NOT our actual max_model_len=8192. That uncapped value flows straight into
# TritonAttnBackend.max_context_len, which directly sizes the EAGLE draft's
# per-decode-batch kv_indices buffer (triton_backend.py init_forward_metadata)
# as speculative_num_steps * batch_size * topk * max_context_len -- ~16x
# (131072/8192) larger than needed, matching the observed ~3.3 GiB allocation
# exactly at the observed ~100-concurrent-request batch sizes. This explains
# why crashes hit at inconsistent step numbers (0, 7, 9) with near-identical
# memory numbers each time: not a leak, a single oversized transient
# allocation whose success depends on the fragmented pool's momentary layout.
# Fixed in sglang_rollout.py by passing context_length=self.config.max_model_len
# (+ the SD verify-token reserve, see below) -- see that call site's comment
# for the full writeup, including why every other context_len consumer in
# sglang was checked and is unaffected for this workload.
#
# Attempt 8 (job 5338906, context_length fix, no SD reserve margin yet):
# rollout ran ~21min before a NEW, different rejection: sglang's own request
# admission check (tokenizer_manager.py _validate_one_request) adds a
# `reserve_input_token_num` margin (= speculative_num_draft_tokens, 48 in our
# config) on top of input+max_new_tokens before comparing to context_len --
# previously invisible when context_len was 131072 (huge accidental slack),
# now exposed because verl's own max_new_tokens computation
# (sglang_rollout.py's `max_new_tokens = min(response_length, max_model_len -
# len(generation_prompt_ids) - 1)`) never accounted for it. Fixed by widening
# context_length by that same reserve when SD is on (8192 -> 8240, ~0.6%,
# negligible next to the buffer this was already shrinking 16x) -- see the
# same call site's comment.
#
# Attempt 9 (job 5353584, both context_length fixes applied): got through
# 3/10 steps cleanly (no OOM, no rejection -- both fixes holding), then hit
# YET ANOTHER 600s collective timeout, this one the same actor-training NCCL
# ALLREDUCE straggler issue as attempt 4 (DP-rank rollout duration variance
# at production scale). The nccl_timeout=1800 override from attempt 4/5
# turned out to not actually be reaching the relevant process group:
# `self.config.get("nccl_timeout", 600)` in fsdp_workers.py reads from
# `self.config` = `config.actor_rollout_ref` as a whole (see
# ray_trainer.py:869,889 `config=self.config.actor_rollout_ref`), a sibling
# key to `.actor`/`.rollout`/`.model` -- NOT `config.trainer`. So
# `+trainer.nccl_timeout=1800` silently set a key nothing reads; the log
# still showed `Timeout(ms)=600000`.
#
# Attempt 10 (job 5358701, `+actor_rollout_ref.nccl_timeout=1800`): failed at
# Hydra parse -- "Could not append... An item is already at
# 'actor_rollout_ref.nccl_timeout'" confirms the path itself was right (the
# key already exists in the actor_rollout_ref struct), just the `+` (append)
# prefix was wrong for an existing key -- `+` is for adding a NEW key,
# `trainer.save_freq` above needs no prefix for the same reason. Corrected to
# a plain `actor_rollout_ref.nccl_timeout=1800` override, no GPU time lost.

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-tlt-norafter-prod
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
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

# production scale (matches run_tlt_flagship_slurm.sh's non-smoke branch),
# step count overridable for shorter/longer probes
TRAIN_PROMPT_BSZ=256; MINI_BSZ=256; N_RESP=5
MAX_STEPS=${MAX_STEPS:-10}
SAVE_FREQ=-1  # metrics-only probe, no checkpoint I/O to keep latency numbers clean
RUN_TAG="tlt-norafter-prod-${SLURM_JOB_ID}"

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

# PYTHONUNBUFFERED: see the attempt-1 note above -- without this, a crash
# loses every step's metrics that hadn't reached a stdio flush yet.
export PYTHONUNBUFFERED=1

# 6h internal cutoff inside the 8h SBATCH allocation, so a hang exits cleanly
# with rc=124 and still leaves time for this script's own teardown/echo to
# run, instead of getting hard-killed by Slurm with no final log line.
set +e
timeout 21600 env \
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
bash examples/grpo_sql_baseline.sh trainer.save_freq=$SAVE_FREQ actor_rollout_ref.nccl_timeout=1800
rc=$?
set -e

echo "=== done (rc=$rc) ==="
# Triage (log is still open for writing from inside this script, so grep it
# after the fact rather than from in here):
#   grep -E '^step:' %x-$SLURM_JOB_ID.out                      # reward / gen length / timing per step
#   grep -o 'accept len: [0-9.]*' %x-$SLURM_JOB_ID.out | sort -u  # SD acceptance distribution
#   grep -o 'gen throughput (token/s): [0-9.]*' %x-$SLURM_JOB_ID.out
exit $rc
