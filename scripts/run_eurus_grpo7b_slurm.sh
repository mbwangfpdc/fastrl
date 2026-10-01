#!/bin/bash
#
# FastRL's own examples/grpo_7B.sh (Qwen2.5-7B base, GRPO on Eurus-2-RL-Data,
# 1 node x 8 GPUs), run on 4 Oscar B200s to compare its rollout/response-length
# dynamics with granular-cais-rl's port. Every training argument is copied from
# grpo_7B.sh; the ONLY changes are:
#   - speculative.enable=false  (measure training dynamics, not the drafter)
#   - data paths -> /users/mborjigi/data/datasets/eurus2_rl (the same
#     PRIME-RL/Eurus-2-RL-Data parquet files the script expects)
#   - trainer.total_training_steps=$STEPS instead of a full epoch
#   - trainer.save_freq=-1 (no checkpoints)
#   - trainer.n_gpus_per_node=4 instead of 8 (batch sizes are global, so the RL
#     problem is unchanged; rollout TP 4 = one engine, Ulysses SP 4 = one DP group)
#
#   sbatch --export=ALL,STEPS=3 scripts/run_eurus_grpo7b_slurm.sh
#
# Per-step metrics: grep '^.*step:[0-9]' the .out log (response_length/*, timing_s/*).
#
#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-eurus-grpo7b
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --constraint=b200
#SBATCH --mem=768g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
REPO=/oscar/data/deeptir/mborjigi/fastrl
cd "$REPO"
STEPS=${STEPS:-3}
echo "node=$(hostname) job=$SLURM_JOB_ID steps=$STEPS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

ulimit -u "$(ulimit -Hu)"
module load cuda/12.9.0-cinr
export RAY_OVERRIDE_RESOURCES="{\"CPU\":$SLURM_CPUS_PER_TASK}"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
# Build the venv on this node if it doesn't exist yet (the CPU queues can be
# hours deep; building here costs ~20-30 min of idle GPU time instead).
if [ ! -x "$REPO/.venv/bin/python" ] || ! "$REPO/.venv/bin/python" -c "import verl, sglang" 2>/dev/null; then
  echo "=== building $REPO/.venv on $(hostname) ==="
  bash "$REPO/scripts/setup_env_core_slurm.sh"
fi
source "$REPO/.venv/bin/activate"
export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}
export TOKENIZERS_PARALLELISM=true NCCL_DEBUG=WARN MKL_SERVICE_FORCE_INTEL=1 PYTHONUNBUFFERED=1

DATA_PATH=/users/mborjigi/data/datasets/eurus2_rl
MODEL_PATH=Qwen/Qwen2.5-7B
SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL
CKPT_PATH=$REPO/output/ckpt
PROJECT_NAME=FastRL
EXPERIMENT_NAME="Qwen2.5-7B-eurus-${SLURM_JOB_ID}"

train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=4
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 32))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

ray stop --force || true
sleep 3

python3 -m verl.trainer.main_fastrl \
    speculative.eagle.spec_model_path=$SPEC_MODEL_PATH \
    speculative.enable=false \
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
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$STEPS
echo "=== done ==="
