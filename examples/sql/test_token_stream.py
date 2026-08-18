# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check the multi-turn token stream shape. CPU only, no GPU/model weights.

The SkyRL-SQL reward parser requires the text after every `</observation>` to
start with `<think>`. verl's default chat-template multi-turn interposes the
literal role word ("assistant") once special tokens are stripped, which fails
that check on *every* multi-turn trajectory -- silently, as an all -1.0 reward.

This asserts that `use_conversation_multi_turn=False` produces one continuous
assistant stream that the parser accepts, and demonstrates that the templated
mode does not.
"""

from __future__ import annotations

import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from skyrl_gym.envs.sql.utils import verify_format_and_extract  # noqa: E402
from verl.workers.rollout.schemas import (  # noqa: E402
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
)

MODEL = os.environ.get("TOKENIZER", "Qwen/Qwen2.5-Coder-7B-Instruct")

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail and not cond else ''}")


def build_request(tokenizer, multi_turn: bool) -> AsyncRolloutRequest:
    messages = [
        {"role": "system", "content": "You are a data science expert."},
        {"role": "user", "content": "Question: how many pigs?"},
    ]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    return AsyncRolloutRequest(
        batch_data_id=0,
        rollout_offset=0,
        request_id="tok-test",
        state=AsyncRolloutRequestStateEnum.PENDING,
        messages=messages,
        tool_schemas=None,
        tools_kwargs={},
        interaction_kwargs={},
        input_ids=ids,
        response_ids=None,
        attention_mask=torch.ones_like(ids),
        response_attention_mask=None,
        response_position_ids=None,
        response_loss_mask=None,
        reward_scores={},
        max_prompt_len=4096,
        max_response_len=4096,
        max_model_len=16384,
        use_inference_chat_template=False,
        use_conversation_multi_turn=multi_turn,
        tokenization_sanity_check_mode="disable",
        processing_class=tokenizer,
    )


def run_episode(tokenizer, multi_turn: bool):
    """Drive two assistant turns with one observation between them."""
    req = build_request(tokenizer, multi_turn)
    prompt_len = req.input_ids.shape[-1]

    req.get_generation_prompt_ids(tokenizer)
    req.add_assistant_message(tokenizer, "<think>first</think><sql>SELECT 1;</sql>")
    req.add_user_message(tokenizer, "\n\n<observation>x\n<reminder>4 turns left</reminder></observation>\n\n")
    req.get_generation_prompt_ids(tokenizer)
    req.add_assistant_message(tokenizer, "<think>second</think><solution>SELECT 1;</solution>")

    response_ids = req.input_ids[0, prompt_len:]
    loss_mask = req.loss_mask[0, prompt_len:]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return req, text, response_ids, loss_mask


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print(f"tokenizer: {MODEL}\n")

    print("1. use_conversation_multi_turn=False (SkyRL convention)")
    req, text, response_ids, loss_mask = run_episode(tokenizer, multi_turn=False)
    ok, _, pred_sql, _ = verify_format_and_extract(text)
    check("reward parser accepts the trajectory", ok is True)
    check("extracted solution SQL", pred_sql == "SELECT 1;", f"got {pred_sql!r}")
    check("no 'assistant' role word leaked into the response", "assistant" not in text)
    check("no 'user' role word leaked into the response", "user" not in text)
    check("<think> follows </observation> directly", "</observation>\n\n<think>" in text)
    check(
        "observation tokens are masked out of the loss",
        int(loss_mask.sum()) < len(loss_mask),
        f"masked={len(loss_mask) - int(loss_mask.sum())}",
    )
    check("lengths stay consistent", req.input_ids.shape[-1] == req.loss_mask.shape[-1] == req.attention_mask.shape[-1])
    print(f"    response[:150]={text[:150]!r}")
    print(f"    tokens={len(response_ids)} trainable={int(loss_mask.sum())}\n")

    print("2. use_conversation_multi_turn=True (verl default) -- expected to fail the parser")
    _, text_mt, response_mt, _ = run_episode(tokenizer, multi_turn=True)
    ok_mt, _, _, _ = verify_format_and_extract(text_mt)
    check("templated mode is rejected by the parser (documents the bug)", ok_mt is False)
    check("templated mode leaks the role word", "assistant" in text_mt)
    print(f"    response[:150]={text_mt[:150]!r}")
    print(f"    tokens={len(response_mt)} (vs {len(response_ids)} continuous)\n")

    print(f"passed {len(_PASS)}/{len(_PASS) + len(_FAIL)}")
    if _FAIL:
        print("FAILED: " + ", ".join(_FAIL))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
