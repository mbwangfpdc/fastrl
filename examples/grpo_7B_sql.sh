#!/usr/bin/env bash
# GRPO on the SkyRL-SQL workload, with FastRL's adaptive speculative decoding.
#
# The multi-turn <sql>/<observation>/<solution> loop is driven by
# verl/interactions/sql_interaction.py, which delegates execution to skyrl_gym's
# SQLEnv. No tools are registered: SkyRL-SQL is a raw-tag protocol, not an
# OpenAI function-calling one, so installing a tool-call parser would change the
# prompt shape and therefore the response-length distribution being measured.
#
# Prepare data first:
#   python examples/sql/prepare_skyrl_sql_data.py \
#       --input  /local_nvme1/mborjigi/data/text2sql-data/train.parquet \
#       --output data/skyrl_sql/train.parquet \
#       --db-path /local_nvme1/mborjigi/data/text2sql-data/data --split train
#   python examples/sql/prepare_skyrl_sql_data.py \
#       --input  /local_nvme1/mborjigi/data/text2sql-data/validation.parquet \
#       --output data/skyrl_sql/validation.parquet \
#       --db-path /local_nvme1/mborjigi/data/text2sql-data/data --split validation
#
# Toggle SD off for the A/B baseline:  SPECULATIVE=false bash examples/grpo_7B_sql.sh
set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export MKL_SERVICE_FORCE_INTEL=1

CKPT_PATH=${CKPT_PATH:-output/fastrl_sql}
PROJECT_NAME=${PROJECT_NAME:-FastRL-SQL}
SPECULATIVE=${SPECULATIVE:-true}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-skyrl-sql-7B-sd${SPECULATIVE}}

# Target policy. SkyRL-SQL-7B is the model the workload was defined against, so
# it produces a representative turn count and response-length distribution.
MODEL_PATH=${MODEL_PATH:-NovaSky-AI/SkyRL-SQL-7B}

# WARNING: EAGLE drafters are trained against one specific target model and are
# sensitive to the prompt prefix (see README). mit-han-lab/Qwen2.5-7B-Eagle-RL
# is trained for Qwen2.5-7B -- pairing it with SkyRL-SQL-7B will accept poorly
# and will understate FastRL's speedup. For a headline number, train a drafter
# for this target with eagle-train/ and point SPEC_MODEL_PATH at it. To smoke
# test the plumbing on a matched pair instead, set
#   MODEL_PATH=Qwen/Qwen2.5-7B SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL
# (accepting that a non-SQL-tuned policy mostly emits format errors).
SPEC_MODEL_PATH=${SPEC_MODEL_PATH:-mit-han-lab/Qwen2.5-7B-Eagle-RL}

DATA_PATH=${DATA_PATH:-data/skyrl_sql}
DB_PATH=${DB_PATH:-/local_nvme1/mborjigi/data/text2sql-data/data}

NGPUS=${NGPUS:-4}
TP=${TP:-2}
ULYSSES=${ULYSSES:-2}

# Mirrors granular-cais-rl's SQL config: 5 generations per prompt, 5 assistant
# turns, 4096-token prompts, temperature 0.6 / top_p 0.95.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-32}
n_resp_per_prompt=${N_RESP:-5}
train_prompt_mini_bsz=${MINI_BSZ:-4}
max_prompt_length=4096
# One knob covers the whole multi-turn response in verl (generated text *and*
# observation tokens), so this is the episode budget, not a per-turn cap.
max_response_length=${MAX_RESPONSE_LENGTH:-12288}
max_model_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

# Env turn budget is 5; give verl one more so the environment, not the rollout
# loop, decides when an episode ends.
max_assistant_turns=${MAX_ASSISTANT_TURNS:-6}

ray stop --force
sleep 3

python3 -m verl.trainer.main_fastrl \
    speculative.enable=${SPECULATIVE} \
    speculative.eagle.spec_model_path=$SPEC_MODEL_PATH \
    speculative.bs_threshold=32 \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/validation.parquet \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ULYSSES} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +actor_rollout_ref.rollout.stop="['</sql>','</solution>']" \
    +actor_rollout_ref.rollout.no_stop_trim=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=examples/sql/interaction_config.yaml \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_assistant_turns} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
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
    trainer.total_epochs=1 \
    "$@"
