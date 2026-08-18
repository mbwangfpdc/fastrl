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
"""Convert the SkyRL-SQL parquet into the layout verl's RLHFDataset expects.

The `prompt` column is copied through untouched -- prompt fidelity is the whole
point, since prompt shape drives the response-length distribution that FastRL's
adaptive speculative decoding is measured against.

Input columns (as shipped for granular-cais-rl):
    prompt, db_id, data, reward_spec{ground_truth}, ...

Output columns (verl):
    prompt        chat messages, unchanged
    data_source   reward_fn_key; only a label here since scoring goes through
                  custom_reward_function
    reward_model  {style, ground_truth}
    extra_info    {index, split, db_id, data, db_path, interaction_kwargs{...}}

Usage:
    python examples/sql/prepare_skyrl_sql_data.py \
        --input  /local_nvme1/mborjigi/data/text2sql-data/train.parquet \
        --output data/skyrl_sql/train.parquet \
        --db-path /local_nvme1/mborjigi/data/text2sql-data/data \
        --split train
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def _ground_truth(row: pd.Series) -> str:
    spec = row.get("reward_spec")
    if isinstance(spec, dict) and "ground_truth" in spec:
        return spec["ground_truth"]
    # Some dumps carry the gold query in a bare `sql` column instead.
    if "sql" in row and isinstance(row["sql"], str):
        return row["sql"]
    raise KeyError("row has neither reward_spec.ground_truth nor sql")


def convert(
    df: pd.DataFrame,
    db_path: str,
    split: str,
    data_source: str,
    max_turns: int,
    default_task: str,
) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        db_id = row["db_id"]
        task = row["data"] if "data" in row and isinstance(row["data"], str) else default_task
        gold = _ground_truth(row)

        # Flat scalars only: this round-trips through parquet and back out of
        # verl's RLHFDataset as a plain dict, which nested structs do not
        # reliably do.
        interaction_kwargs = {
            "name": "sql",
            "db_id": db_id,
            "ground_truth": gold,
            "data": task,
            "max_turns": max_turns,
        }

        rows.append(
            {
                "prompt": row["prompt"],
                "data_source": data_source,
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {
                    "index": int(i),
                    "split": split,
                    "db_id": db_id,
                    "data": task,
                    "db_path": db_path,
                    "interaction_kwargs": interaction_kwargs,
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source parquet")
    ap.add_argument("--output", required=True, help="destination parquet")
    ap.add_argument("--db-path", required=True, help="root of the text2sql database tree")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-source", default="skyrl_sql", help="verl reward_fn_key label")
    ap.add_argument("--max-turns", type=int, default=5, help="assistant-turn budget per sample")
    ap.add_argument("--default-task", default="synsql", help="fallback when a row omits `data`")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N rows")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.head(args.limit)

    out = convert(df, args.db_path, args.split, args.data_source, args.max_turns, args.default_task)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
