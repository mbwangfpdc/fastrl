#!/usr/bin/env python
"""Generate a full-fidelity SkyRL-SQL trajectory corpus for EAGLE drafter training.

EAGLE learns to predict *the target model's own continuations*, and FastRL's
README is explicit that the drafter is prefix-sensitive. So the corpus has to be
this target generating on these prompts, in this protocol -- not a generic chat
mix.

Why not reuse the reward-function dumps in `output/rollouts_*.jsonl`: they
truncate the text at 4000 chars and omit the prompt entirely, so they cannot be
reassembled into (prompt, continuation) pairs.

This runs the SkyRL-SQL agent loop directly against an offline sglang engine and
the same `SQLInteraction` the trainer uses, so the emitted text is byte-identical
in protocol to what RL rollouts produce. Output is a parquet with a `messages`
column, which `eagle_datagen.py` consumes directly.

Deliberately modest about GPUs: generation only, so it runs fine on 1-2 cards
while the rest of the box is busy.

    python examples/sql/build_eagle_corpus.py \
        --data data/skyrl_sql/train.parquet \
        --out  data/eagle_corpus/train.parquet \
        --samples-per-prompt 3 --tp 2

NB: run it with fastrl's venv (it needs sglang + skyrl_gym), NOT eagle-train's.
The corpus is plain parquet, so the two environments only meet on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The protocol's stop tags: generation halts at the end of each SQL block or the
# final solution, exactly as in training.
STOP_TAGS = ["</sql>", "</solution>"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/skyrl_sql/train.parquet")
    p.add_argument("--out", default="data/eagle_corpus/train.parquet")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--db-root", default="/local_nvme1/mborjigi/data/text2sql-data/data")
    p.add_argument("--samples-per-prompt", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="cap prompts (smoke tests)")
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=64,
                   help="in-flight trajectories; each holds one SQLite env")
    p.add_argument("--mem-fraction", type=float, default=0.75)
    return p.parse_args()


async def run_one(engine, interaction, tokenizer, row, sample_idx, args):
    """Replay the SQL loop once; return a messages record or None."""
    from verl.interactions.sql_interaction import _truncate_at_stop_tag

    prompt_msgs = [dict(m) for m in row["prompt"]]
    ik = dict(row["extra_info"]["interaction_kwargs"])
    instance_id = f"{row['extra_info']['index']}-{sample_idx}"

    await interaction.start_interaction(instance_id=instance_id, **ik)

    # Continuous-stream protocol (`use_conversation_multi_turn=False`): the whole
    # trajectory is ONE assistant turn -- model text and injected observations
    # concatenated, with no per-turn role wrapping. Matching that here is the
    # point; wrapping each turn would train the drafter on a token stream the
    # policy never actually sees.
    prefix = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
    stream = ""

    for _ in range(args.max_turns):
        out = await engine.async_generate(
            prompt=prefix + stream,
            sampling_params={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "stop": STOP_TAGS,
                "no_stop_trim": True,
            },
        )
        action = out["text"]
        if not action:
            break
        stream += action

        # skyrl_gym asserts on trailing text after </sql>, so cut at the tag.
        truncated, _ = _truncate_at_stop_tag(action)
        try:
            done, observation, _reward, _info = await interaction.generate_response(
                instance_id, [{"role": "assistant", "content": truncated}]
            )
        except Exception:
            return None

        if done:
            break
        if observation:
            stream += observation

    try:
        await interaction.finalize_interaction(instance_id)
    except Exception:
        pass

    if not stream.strip():
        return None
    return {"messages": prompt_msgs + [{"role": "assistant", "content": stream}]}


async def main_async(args):
    import sglang as sgl
    from transformers import AutoTokenizer
    from verl.interactions.sql_interaction import SQLInteraction

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"prompts: {len(df)} x {args.samples_per_prompt} samples "
          f"= {len(df) * args.samples_per_prompt} trajectories", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    interaction = SQLInteraction({"db_path": args.db_root, "max_turns": args.max_turns})

    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp,
        mem_fraction_static=args.mem_fraction,
        # flashinfer's JIT does not build on this box; triton is the working backend.
        attention_backend="triton",
        log_level="warning",
    )

    # Build every (prompt, sample) job up front and run them with a bounded
    # concurrency. Each job owns its own instance_id -- and therefore its own
    # SQLEnv/SQLite handle -- so cross-prompt concurrency is safe, and it is what
    # keeps the engine busy: one prompt at a time would leave it mostly idle
    # between the short turns of the agent loop.
    jobs = [(row, s) for _, row in df.iterrows() for s in range(args.samples_per_prompt)]
    sem = asyncio.Semaphore(args.concurrency)
    records, failures, done = [], 0, 0

    async def guarded(row, s):
        async with sem:
            return await run_one(engine, interaction, tokenizer, row, s, args)

    try:
        pending = [asyncio.create_task(guarded(row, s)) for row, s in jobs]
        for fut in asyncio.as_completed(pending):
            r = await fut
            done += 1
            if r is None:
                failures += 1
            else:
                records.append(r)
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} trajectories ({failures} failed)", flush=True)
    finally:
        engine.shutdown()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out)
    print(f"\nwrote {len(records)} trajectories to {out} ({failures} failed)")

    if records:
        sample = records[0]["messages"][-1]["content"]
        print(f"sample assistant stream: {len(sample)} chars, "
              f"{sample.count('<sql>')} <sql>, {sample.count('<observation>')} <observation>, "
              f"solution={'<solution>' in sample}")


if __name__ == "__main__":
    a = parse_args()
    os.environ.setdefault("HF_HOME", "/local_nvme0/mborjigi/hf")
    asyncio.run(main_async(a))
