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
"""Training reward for the SkyRL-SQL workload.

Wire this up with::

    custom_reward_function.path=examples/sql/sql_reward.py
    custom_reward_function.name=compute_score

Why a reward function *as well as* the interaction: the per-turn reward the
interaction returns is carried in the rollout's ``reward_scores`` but is never
read back by any of verl's reward managers -- the training signal comes from
``compute_score`` over the decoded response. That split is convenient here,
because it means a trajectory truncated by ``max_assistant_turns`` (so the env
never reached its terminal branch) still gets scored correctly.

Scoring is skyrl_gym's own ``calculate_reward_single``, so the semantics match
the SkyRL-SQL workload exactly:

    -1.0  format error (no/!malformed <solution>, stray tags, junk after
          </observation>)
     1.0  predicted SQL result set matches gold
     0.0  otherwise

Note this is the strict, column-order-sensitive ``frozenset(fetchall())``
comparison used for *training* reward -- it scores a few points below the
OmniSQL benchmark metric, so do not compare it directly to published exec-acc.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from verl.interactions.sql_interaction import resolve_db_file

logger = logging.getLogger(__name__)

# Both the predicted and the gold query are executed under this timeout.
_DEFAULT_TIMEOUT = int(os.getenv("SKYRL_SQL_REWARD_TIMEOUT", "30"))


def compute_score(
    data_source: Optional[str] = None,
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    """Score one SkyRL-SQL trajectory.

    ``solution_str`` is the decoded response, which in a multi-turn rollout
    interleaves generated text with the observation tokens -- the same shape as
    the concatenated chat history skyrl_gym scores, so it can be passed straight
    through.
    """
    from skyrl_gym.envs.sql.utils import calculate_reward_single

    extra_info = extra_info or {}

    db_root = extra_info.get("db_path") or os.getenv("SKYRL_SQL_DB_PATH")
    db_id = extra_info.get("db_id")
    task = extra_info.get("data") or "synsql"

    if not db_root or not db_id:
        # Misconfigured data would otherwise silently train on all-zero reward.
        raise ValueError(
            "SkyRL-SQL reward needs `db_id` and a database root; set extra_info.db_id and "
            "either extra_info.db_path or $SKYRL_SQL_DB_PATH. "
            f"got db_id={db_id!r} db_path={db_root!r}"
        )

    db_file = resolve_db_file(db_root, task, db_id)

    try:
        score = float(calculate_reward_single(solution_str, ground_truth, db_file, timeout=_DEFAULT_TIMEOUT))
    except Exception:
        logger.exception("SkyRL-SQL reward failed for db_id=%s; scoring 0.0", db_id)
        score = 0.0

    _maybe_dump(solution_str, score, db_id, task)
    return score


# --- rollout inspection -------------------------------------------------------
# Set SKYRL_SQL_DUMP_PATH to append scored trajectories as JSONL. This is the
# cheapest way to check that the generation harness is actually turning the loop
# (observations injected, several turns) rather than emitting one-and-done
# answers -- a failure mode that still produces plausible-looking reward curves.
_DUMP_PATH = os.getenv("SKYRL_SQL_DUMP_PATH")
_DUMP_LIMIT = int(os.getenv("SKYRL_SQL_DUMP_N", "200"))
_dumped = 0


def _maybe_dump(solution_str: str, score: float, db_id: str, task: str) -> None:
    global _dumped
    if not _DUMP_PATH or _dumped >= _DUMP_LIMIT:
        return
    import json

    text = solution_str or ""
    record = {
        "score": score,
        "db_id": db_id,
        "task": task,
        "n_sql": text.count("<sql>"),
        "n_observation": text.count("<observation>"),
        "n_think": text.count("<think>"),
        "has_solution": "<solution>" in text,
        "chars": len(text),
        "text": text[:4000],
    }
    try:
        with open(_DUMP_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        _dumped += 1
    except Exception:
        logger.exception("failed writing rollout dump")
