#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VIRTUAL_ENV="$PWD/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export HF_HOME=/local_nvme0/mborjigi/hf
export CUDA_HOME=/usr/local/cuda; export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# Tengyunw is full MHA (kv_heads 28 vs 4), so its draft KV is ~7x heavier --
# hence a lower engine fraction than the GQA drafters got.
timeout 1200 python scripts/bench_speculative_decoding.py \
    --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --data_dir data/eagle_corpus/bench_prompts.json \
    --spec_algorithm EAGLE --eagle_path Tengyunw/qwen_2.5_7b_instruct_eagle2_v0 \
    --speculative_num_steps 8 --speculative_eagle_topk 4 \
    --speculative_num_draft_tokens 48 --attention_backend triton --max_bs 8 \
    --context_length 8192 --mem_fraction_static 0.5 \
    > output/drafter_bench/tengyunw.log 2>&1
echo "rc=$?"; grep -Ei "accept" output/drafter_bench/tengyunw.log | tail -3
