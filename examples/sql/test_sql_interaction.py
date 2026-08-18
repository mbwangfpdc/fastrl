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
"""CPU-only checks for the SkyRL-SQL interaction. No GPU, no model.

    python examples/sql/test_sql_interaction.py \
        --dataset /local_nvme1/mborjigi/data/text2sql-data/train.parquet \
        --db-path /local_nvme1/mborjigi/data/text2sql-data/data

Drives SQLInteraction the way verl's rollout does -- start_interaction, then
generate_response once per assistant turn -- against a real SQLite database, and
checks the observation format, the terminal branch, the turn budget, the
format-error reward and the training reward function.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verl.interactions.sql_interaction import SQLInteraction, resolve_db_file  # noqa: E402

_PASS, _FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail and not cond else ''}")


def pick_row(dataset: str, db_path: str) -> pd.Series:
    """First row whose SQLite file actually exists on disk."""
    df = pd.read_parquet(dataset)
    for _, row in df.head(200).iterrows():
        task = row["data"] if isinstance(row.get("data"), str) else "synsql"
        if os.path.exists(resolve_db_file(db_path, task, row["db_id"])):
            return row
    raise SystemExit(f"no usable row found in the first 200 rows of {dataset}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/local_nvme1/mborjigi/data/text2sql-data/train.parquet")
    ap.add_argument("--db-path", default="/local_nvme1/mborjigi/data/text2sql-data/data")
    args = ap.parse_args()

    row = pick_row(args.dataset, args.db_path)
    db_id = row["db_id"]
    task = row["data"] if isinstance(row.get("data"), str) else "synsql"
    gold = row["reward_spec"]["ground_truth"]
    print(f"sample: db_id={db_id} task={task}\n")

    kwargs = {"name": "sql", "db_id": db_id, "ground_truth": gold, "data": task, "max_turns": 5}
    interaction = SQLInteraction({"db_path": args.db_path, "max_turns": 5})

    # ---- 1. an intermediate <sql> turn returns a formatted observation --------
    print("1. intermediate <sql> turn")
    iid = await interaction.start_interaction("req-1", **kwargs)
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": f"<think>t</think><sql>{gold}</sql>"}]
    done, obs, reward, meta = await interaction.generate_response(iid, msgs, **kwargs)
    check("does not terminate on <sql>", done is False)
    check("observation is wrapped in <observation>", "<observation>" in obs and "</observation>" in obs)
    check("observation carries the turn <reminder>", "<reminder>" in obs)
    check("intermediate reward is 0", reward == 0.0, f"got {reward}")
    check("metadata reports turn 1", meta.get("turns") == 1, f"got {meta.get('turns')}")
    print(f"    obs[:110]={obs[:110]!r}\n")

    # ---- 2. <solution> terminates and scores ---------------------------------
    print("2. terminal <solution> turn")
    msgs.append({"role": "user", "content": obs})
    msgs.append({"role": "assistant", "content": f"<think>t</think><solution>{gold}</solution>"})
    done, obs2, reward, meta = await interaction.generate_response(iid, msgs, **kwargs)
    check("terminates on <solution>", done is True)
    check("gold SQL scores 1.0", reward == 1.0, f"got {reward}")
    check("env released after terminal turn", "req-1" not in interaction._instance_dict)
    print()

    # ---- 3. malformed output is a format error (-1.0) ------------------------
    print("3. format error")
    iid = await interaction.start_interaction("req-2", **kwargs)
    bad = [{"role": "assistant", "content": "I think the answer is probably SELECT 1"}]
    done, _, reward, _ = await interaction.generate_response(iid, bad, **kwargs)
    check("no <solution> keeps the episode alive", done is False)
    for turn in range(2, 6):  # exhaust the 5-turn budget
        done, _, reward, meta = await interaction.generate_response(iid, bad, **kwargs)
    check("turn budget terminates the episode", done is True, f"turns={meta.get('turns')}")
    check("format error scores -1.0", reward == -1.0, f"got {reward}")
    print()

    # ---- 4. wrong-but-well-formed SQL scores 0.0 ----------------------------
    print("4. wrong answer")
    iid = await interaction.start_interaction("req-3", **kwargs)
    wrong = [{"role": "assistant", "content": "<think>t</think><solution>SELECT 1 AS x</solution>"}]
    done, _, reward, _ = await interaction.generate_response(iid, wrong, **kwargs)
    check("terminates", done is True)
    check("wrong SQL scores 0.0", reward == 0.0, f"got {reward}")
    print()

    # ---- 5. the training reward function agrees -----------------------------
    print("5. training reward function")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sql_reward import compute_score  # noqa: E402

    extra = {"db_id": db_id, "data": task, "db_path": args.db_path}
    completion = f"<think>t</think><sql>{gold}</sql>\n\n<observation>x</observation>\n\n<think>t</think><solution>{gold}</solution>"
    check("gold trajectory scores 1.0", compute_score("skyrl_sql", completion, gold, extra) == 1.0)
    check(
        "unformatted trajectory scores -1.0",
        compute_score("skyrl_sql", "no tags at all", gold, extra) == -1.0,
    )
    check(
        "wrong solution scores 0.0",
        compute_score("skyrl_sql", "<think>t</think><solution>SELECT 1 AS x</solution>", gold, extra) == 0.0,
    )
    print()

    # ---- 6. instance table stays bounded ------------------------------------
    print("6. env table is bounded")
    small = SQLInteraction({"db_path": args.db_path, "max_turns": 5, "max_instances": 4})
    for n in range(10):
        await small.start_interaction(f"leak-{n}", **kwargs)
    check("evicts beyond max_instances", len(small._instance_dict) == 4, f"got {len(small._instance_dict)}")
    print()

    print(f"passed {len(_PASS)}/{len(_PASS) + len(_FAIL)}")
    if _FAIL:
        print("FAILED: " + ", ".join(_FAIL))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
