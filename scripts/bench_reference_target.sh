#!/usr/bin/env bash
# Is our 2.9 accept length "good"? The repo publishes no number, so measure
# mit-han-lab's drafter against ITS OWN target family (Qwen2.5-7B-Instruct)
# on the identical harness/prompts. That isolates the model-match variable:
#   - on OUR target it scored 1.178 (mismatched)
#   - on its own target it should score near whatever the authors intended
# Caveat: the prompts are SkyRL-SQL, not their domain, so this bounds the
# model-match effect rather than reproducing a paper number.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VIRTUAL_ENV="$PWD/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export HF_HOME=/local_nvme0/mborjigi/hf
export CUDA_HOME=/usr/local/cuda; export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
mkdir -p output/drafter_bench
run() {
  echo "########## $1  (target=$2)"
  timeout 1500 python scripts/bench_speculative_decoding.py \
      --model_path "$2" --data_dir data/eagle_corpus/bench_prompts.json \
      --spec_algorithm "$3" ${4:+--eagle_path "$4"} \
      --speculative_num_steps 8 --speculative_eagle_topk 4 \
      --speculative_num_draft_tokens 48 --attention_backend triton --max_bs 8 \
      --context_length 8192 --mem_fraction_static 0.6 \
      > "output/drafter_bench/$1.log" 2>&1
  echo "rc=$?"; grep -Ei "accept" "output/drafter_bench/$1.log" | tail -3
}
# mit-han-lab drafter on its own target family
run mhl_on_own_target Qwen/Qwen2.5-7B-Instruct EAGLE mit-han-lab/Qwen2.5-7B-Eagle-RL
# SD-off on the same target, for that model's 1.0 baseline
run sdoff_qwen_instruct Qwen/Qwen2.5-7B-Instruct None ""
echo "########## done"
