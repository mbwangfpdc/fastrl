#!/usr/bin/env bash
# Phase 4 (offline): acceptance length + standalone speedup per drafter, on
# held-out SkyRL-SQL prompts. This is how the checkpoint gets chosen -- the
# trainer logs only value/prob losses, so a falling loss proves nothing about
# acceptance.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VIRTUAL_ENV="$PWD/.venv"        # fastrl's venv: sglang lives here
export PATH="$VIRTUAL_ENV/bin:$PATH"
export HF_HOME=${HF_HOME:-/local_nvme0/mborjigi/hf}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}   # nvcc needed for CUDA-graph capture
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

PROMPTS=data/eagle_corpus/bench_prompts.json
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
OUT=output/drafter_bench
mkdir -p $OUT

run() {  # name algo eagle_path mem
  echo "########## $1"
  timeout 1200 "$VIRTUAL_ENV/bin/python" scripts/bench_speculative_decoding.py \
      --model_path "$MODEL" --data_dir "$PROMPTS" \
      --spec_algorithm "$2" ${3:+--eagle_path "$3"} \
      --speculative_num_steps 8 --speculative_eagle_topk 4 \
      --speculative_num_draft_tokens 48 \
      --attention_backend triton --max_bs 8 \
      --context_length 8192 --mem_fraction_static "$4" \
      > "$OUT/$1.log" 2>&1
  echo "rc=$?"
  grep -Ei "accept|speed|throughput|tokens/s|elapsed" "$OUT/$1.log" | tail -6
}

run sd_off       None   ""                                          0.6
run mit_han_lab  EAGLE  mit-han-lab/Qwen2.5-7B-Eagle-RL             0.6
run ours_warm    EAGLE  /local_nvme1/mborjigi/eagle-drafters/sql-warm-e3  0.6
run ours_cold    EAGLE  /local_nvme1/mborjigi/eagle-drafters/sql-cold-e3  0.6
echo "########## done"
