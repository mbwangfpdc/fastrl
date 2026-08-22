#!/usr/bin/env bash
# GRPO on the SkyRL-SQL workload, mirroring granular-cais-rl's sql_baseline.toml
# so the two systems can be compared on the same workload.
#
#   SPECULATIVE=true   FastRL adaptive speculative decoding on  (default)
#   SPECULATIVE=false  baseline for the A/B
#
# Mapping from sql_baseline.toml:
#   model_name Qwen/Qwen2.5-Coder-7B-Instruct   -> actor_rollout_ref.model.path
#   lr 1e-6 / wd 0.01 / max_grad_norm 0.5       -> actor.optim.* , actor.grad_clip
#   warmup_steps 0 / lr_schedule constant       -> lr_warmup_steps 0 / warmup_style constant
#   max_steps 35                                -> trainer.total_training_steps
#   entropy_coef 0.0                            -> actor.entropy_coeff
#   batch 256 / mini 256 / num_generations 5    -> train_batch_size / ppo_mini_batch_size / rollout.n
#   temp .6 / top_p .95 / top_k -1              -> rollout.temperature / top_p / top_k
#   max_prompt_len 4096                         -> data.max_prompt_length 4096
#   max_generate_length 3000                    -> data.max_response_length (PER-TURN cap)
#   max_input_length 8192                       -> rollout.max_model_len (trajectory envelope)
#   drop_zero_signal true                       -> algorithm.drop_zero_signal
#   reduce_dtype bfloat16                       -> fsdp_config.mixed_precision.reduce_dtype
#   max_turns 5                                 -> env max_turns, verl max_assistant_turns 6
#   end_tags ["</sql>","</solution>"]           -> rollout.stop  (+ no_stop_trim)
#   use_conversation_multi_turn false           -> multi_turn.use_conversation_multi_turn=False
#   token_budget 24000                          -> ppo_max_token_len_per_gpu
#   vllm_tp_size 1, gpus 0-3                    -> tensor_model_parallel_size 1, n_gpus_per_node 4
#   vllm_enforce_eager true                     -> rollout.enforce_eager (-> sglang disable_cuda_graph)
#   bfloat16                                    -> verl default bf16
#   (no KL term in granular)                    -> use_kl_loss=False, use_kl_in_reward=False
set -euo pipefail

# Pin this repo's venv. Shells on this box export granular-cais-rl's
# VIRTUAL_ENV, so a bare `python3`/`ray` picks up THAT environment and dies
# deep inside sgl_kernel with an unrelated-looking arch error.
FASTRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV="${FASTRL_VENV:-$FASTRL_ROOT/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
PY="$VIRTUAL_ENV/bin/python"
echo "==> interpreter: $PY"

# sglang JIT-compiles kernels when capturing CUDA graphs, which the EAGLE
# speculative-decoding path needs. nvcc lives at /usr/local/cuda but is not on
# PATH by default here; without it SD dies with "Could not find nvcc".
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"

export TOKENIZERS_PARALLELISM=true

# Root of the text2sql database tree, consumed by examples/sql/interaction_config.yaml
# via ${oc.env:SKYRL_SQL_DB_PATH}. Node-specific, so pick the first known root that
# actually exists here rather than hardcoding one machine's layout in the config.
if [ -z "${SKYRL_SQL_DB_PATH:-}" ]; then
  for _cand in /local_nvme1/mborjigi/data/text2sql-data/data \
               /users/mborjigi/data/datasets/skyrl_sql/data \
               /oscar/scratch/mborjigi/data/text2sql-data/data; do
    if [ -d "$_cand" ]; then SKYRL_SQL_DB_PATH="$_cand"; break; fi
  done
fi
if [ -z "${SKYRL_SQL_DB_PATH:-}" ]; then
  echo "ERROR: no text2sql database root found. Set SKYRL_SQL_DB_PATH=<dir holding spider/, bird/, SynSQL-2.5M/>." >&2
  exit 1
fi
export SKYRL_SQL_DB_PATH
echo "==> SKYRL_SQL_DB_PATH: $SKYRL_SQL_DB_PATH"
export NCCL_DEBUG=WARN
export MKL_SERVICE_FORCE_INTEL=1
export HF_HOME=${HF_HOME:-/local_nvme0/mborjigi/hf}
# NB: never set HF_HUB_OFFLINE=1 -- sglang probes for an optional
# hf_quant_config.json and offline mode turns that 404 into a fatal error.
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. It would help
# the rollout/training allocation churn, but torch_memory_saver -- the mechanism
# sglang uses to release and resume the engine's memory between phases -- refuses
# to initialise under expandable segments ("TorchMemorySaver is disabled for the
# current process"), which kills every worker at startup.

# Defaults to FALSE because it is measurably slower at this workload's batch
# size. Matched A/B at 256x5, gpu_mem_util 0.5, step 1:
#   SD off  gen 269s  step 1536s        SD on  gen 402s  step 1668s (step 2 1935s)
# FastRL's SD is *adaptive*: scheduler.py disables speculation whenever the
# running batch exceeds speculative.bs_threshold (32). At 256x5 there are ~1280
# concurrent requests, so speculation only engages on the tail -- while enabling
# it forces disable_overlap_schedule=True and cuda_graph_max_bs=32 engine-wide
# (sglang_rollout.py speculative_args), which taxes the dominant high-batch
# phase. Set SPECULATIVE=true to reproduce the A/B.
SPECULATIVE=${SPECULATIVE:-false}
CKPT_PATH=${CKPT_PATH:-output/fastrl_sql}
PROJECT_NAME=${PROJECT_NAME:-FastRL-SQL}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-baseline-sd${SPECULATIVE}}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
SPEC_MODEL_PATH=${SPEC_MODEL_PATH:-mit-han-lab/Qwen2.5-7B-Eagle-RL}
DATA_PATH=${DATA_PATH:-data/skyrl_sql}
TRAIN_FILE=${TRAIN_FILE:-$DATA_PATH/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_PATH/validation.parquet}
# Set false only for cross-system A/Bs, where another framework has to walk the
# same rows in the same order (SequentialSampler); see
# granular-cais-rl/engine_ab_sglang_vs_vllm_plan.md.
DATA_SHUFFLE=${DATA_SHUFFLE:-True}

NGPUS=${NGPUS:-4}
TP=${TP:-1}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-256}
train_prompt_mini_bsz=${MINI_BSZ:-256}
n_resp_per_prompt=${N_RESP:-5}
max_steps=${MAX_STEPS:-35}

max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
# PER-TURN generation cap, matching granular's max_generate_length. In verl
# multi-turn this bounds each engine call, NOT the trajectory: measured on a
# 3150-trajectory trace with this at 3000, max single-turn decode was exactly
# 3000 while max TOTAL decode reached 4764 (verl only warns past it). The old
# default of 4096 came from mapping granular's max_input_length here, which
# conflated the envelope with the per-turn cap and handed FastRL 36% more
# generation budget per turn.
max_response_length=${MAX_RESPONSE_LENGTH:-3000}
# Trajectory envelope, matching granular's max_input_length. Set independently
# rather than as prompt+response: granular reproduced SkyRL's results at 8192,
# and coupling it to the (now smaller) per-turn cap would shrink it to 7096.
max_model_len=${MAX_MODEL_LEN:-8192}
# sql_baseline.toml uses 24000, but that is granular's packing envelope. verl
# materialises full (T x 152k) logits per microbatch and upcasts for log-softmax,
# so 24000 OOMs a 44GB card. Lowering it only changes microbatch splitting
# (gradient accumulation) -- the gradient is unchanged.
#
# verl asserts ppo_max_token_len_per_gpu * ulysses_sp >= max_seq_len (8192 here),
# so the budget cannot simply be lowered on its own. Ulysses SP=2 shards each
# microbatch's sequence across 2 GPUs, so 4096 * 2 clears the assert while each
# GPU still only materialises 4096 tokens of logits -- half the peak that OOMed.
token_budget=${TOKEN_BUDGET:-4096}
ulysses=${ULYSSES:-2}

# 0.6 measured best at the real 256x5 batch (step 1: 1399s @0.6 vs 1536s @0.5;
# 0.7 gave 1486/1388s, i.e. 0.6-0.7 are equivalent within noise). At 256x5 the
# engine serves ~1280 concurrent requests, so KV capacity -- not the training
# side -- is what generation is starved of. NB 0.5 is required if speculative
# decoding is on with an MHA drafter (see SPECULATIVE above).
gpu_mem_util=${GPU_MEM_UTIL:-0.6}
max_assistant_turns=${MAX_ASSISTANT_TURNS:-6}

# --- execution knobs (do NOT change the gradient; safe to tune for throughput) ---
# Measured decomposition of a 445s step at the defaults below:
#   update_actor 235s (53%) | old_log_prob 75s (17%) | gen 135s (30%, incl 17s reshard)
# So the training side is ~70% of the step and these knobs matter more than SD,
# which can only touch the ~116s of generate_sequences.
param_offload=${PARAM_OFFLOAD:-True}          # shuffles 7B params CPU<->GPU every step
optim_offload=${OPTIM_OFFLOAD:-True}          # ditto for Adam moments
grad_ckpt=${GRAD_CKPT:-True}                  # recomputes the forward inside backward
reshard_after_fwd=${RESHARD_AFTER_FWD:-True}  # False keeps params gathered for backward
fwd_prefetch=${FWD_PREFETCH:-False}
wsync_bucket=${WSYNC_BUCKET:-512}             # weight-sync bucket MB (part of `reshard`)
multi_stage_wake=${MULTI_STAGE_WAKE:-False}
bs_threshold=${BS_THRESHOLD:-32}              # SD engages only once the running batch
                                              # drains below this for 10 consecutive
                                              # decode batches (adaptive SD)

"$VIRTUAL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 3

"$PY" -m verl.trainer.main_fastrl \
    speculative.enable=${SPECULATIVE} \
    speculative.eagle.spec_model_path=$SPEC_MODEL_PATH \
    speculative.bs_threshold=${bs_threshold} \
    speculative.train.enable_drafter_training=${DRAFTER_TRAINING:-False} \
    speculative.train.collect_hidden_states_from_sgl=${DRAFTER_COLLECT_SGL:-False} \
    speculative.train.training_interval_steps=${DRAFTER_INTERVAL:-10} \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.shuffle=${DATA_SHUFFLE} \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=examples/sql/sql_reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${token_budget} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${token_budget} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ulysses} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${grad_ckpt} \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optim_offload} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${reshard_after_fwd} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=${fwd_prefetch} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=${wsync_bucket} \
    actor_rollout_ref.rollout.multi_stage_wake_up=${multi_stage_wake} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False} \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${ATTN_BACKEND:-triton} \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_len} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +actor_rollout_ref.rollout.stop="['</sql>','</solution>']" \
    +actor_rollout_ref.rollout.no_stop_trim=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.use_conversation_multi_turn=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=examples/sql/interaction_config.yaml \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_assistant_turns} \
    algorithm.adv_estimator=grpo \
    algorithm.drop_zero_signal=${DROP_ZERO_SIGNAL:-True} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS:-40} \
    trainer.total_training_steps=${max_steps} \
    "$@"
