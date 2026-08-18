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
"""SkyRL-SQL multi-turn interaction.

The SkyRL-SQL workload does *not* speak verl's OpenAI-style tool-call protocol.
It speaks raw tags: the policy emits ``<sql>...</sql>``, the environment replies
with ``<observation>...</observation>``, and the episode ends on
``<solution>...</solution>`` or when the turn budget runs out.

verl's rollout reaches the ``INTERACTING`` state precisely when no tool call was
parsed out of the assistant message, so an Interaction -- not a Tool -- is the
seam that leaves the protocol byte-identical to the SkyRL-SQL data. Registering
tools instead would inject tool schemas into the chat template and require JSON
function calls, changing prompt shape and response-length distribution, i.e.
exactly the long-tail profile FastRL's adaptive speculative decoding targets.

Execution is delegated to skyrl_gym's ``SQLEnv`` so observation formatting
(including the ``<reminder>`` suffix and 50-row truncation), termination and
reward are the same code paths the SkyRL-SQL numbers were produced with.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Mirrors the task -> database sub-directory mapping inside skyrl_gym's SQLEnv.
# Kept here (rather than imported) so the reward function can resolve the same
# path without constructing an env.
_TASK_DB_SUBDIR = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",
    "bird": "bird/train/train_databases",
}


def resolve_db_file(db_root: str, task: str, db_id: str) -> str:
    """Absolute path of the SQLite file for ``db_id`` under ``db_root``."""
    try:
        subdir = _TASK_DB_SUBDIR[task]
    except KeyError as e:
        raise NotImplementedError(
            f"unknown text2sql task {task!r}; expected one of {sorted(_TASK_DB_SUBDIR)}"
        ) from e
    return os.path.join(db_root, subdir, db_id, db_id + ".sqlite")


def _last_assistant_content(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            return item.get("content") or ""
    return ""


# The stop strings the policy is sampled with. sglang (like vLLM) can emit a few
# extra tokens past a stop string when `no_stop_trim=True`; skyrl_gym 0.3.0
# asserts on any such trailing text, so trim to the first stop tag before the env
# sees the action. Only the env's view is trimmed -- the trained token stream
# keeps whatever was generated, matching granular-cais-rl.
_STOP_TAGS = ("</sql>", "</solution>")


def _truncate_at_stop_tag(action: str) -> tuple[str, str]:
    """Return (action up to and including the earliest stop tag, trailing text)."""
    cut = None
    for tag in _STOP_TAGS:
        idx = action.find(tag)
        if idx != -1:
            end = idx + len(tag)
            if cut is None or end < cut:
                cut = end
    if cut is None:
        return action, ""
    return action[:cut], action[cut:]


class SQLInteraction(BaseInteraction):
    """Drives the SkyRL-SQL ``<sql>``/``<observation>``/``<solution>`` loop.

    Config keys:
        db_path:       root of the text2sql database tree (the directory that
                       contains ``SynSQL-2.5M/``, ``spider/``, ``bird/``).
        max_turns:     default assistant-turn budget when a sample does not
                       carry its own (skyrl_gym's own default is 5).
        default_task:  fallback dataset family when a sample omits ``data``.
        max_instances: bound on retained env instances (see ``_remember``).
    """

    # Process-wide count of generations that ran past a stop string.
    _trailing_stop_text = 0

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.db_root = config.get("db_path")
        if not self.db_root:
            raise ValueError("SQLInteraction requires `db_path` (root of the text2sql database tree)")
        self.default_max_turns = int(config.get("max_turns", 5))
        self.default_task = config.get("default_task", "synsql")
        self.max_instances = int(config.get("max_instances", 65536))
        self._instance_dict: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _remember(self, instance_id: str, entry: dict[str, Any]) -> None:
        """Store an env, evicting the oldest if we are over budget.

        verl's rollout never calls ``finalize_interaction``, and a request that
        is cut short by ``max_assistant_turns`` or by the model running out of
        context never reaches our ``done`` branch -- so without a bound this
        dict grows for the lifetime of the trainer.
        """
        self._instance_dict[instance_id] = entry
        self._instance_dict.move_to_end(instance_id)
        while len(self._instance_dict) > self.max_instances:
            stale_id, stale = self._instance_dict.popitem(last=False)
            logger.warning("evicting un-finalized SQL interaction %s", stale_id)
            self._close(stale)

    @staticmethod
    def _close(entry: dict[str, Any]) -> None:
        env = entry.get("env")
        if env is not None:
            try:
                env.close()
            except Exception:  # pragma: no cover - close is best effort
                logger.exception("error closing SQLEnv")

    def _release(self, instance_id: str) -> None:
        entry = self._instance_dict.pop(instance_id, None)
        if entry is not None:
            self._close(entry)

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs: Any) -> str:
        # Imported lazily so that importing verl.interactions does not hard-require
        # skyrl_gym for users who never run the SQL workload.
        from skyrl_gym.envs.sql.env import SQLEnv, Text2SQLEnvConfig

        if instance_id is None:
            instance_id = str(uuid4())

        db_id = kwargs.get("db_id")
        ground_truth = kwargs.get("ground_truth")
        if db_id is None or ground_truth is None:
            raise ValueError(
                "SQL interaction_kwargs must carry `db_id` and `ground_truth`; got keys "
                f"{sorted(kwargs)}"
            )

        extras = {
            "db_id": db_id,
            "reward_spec": {"ground_truth": ground_truth},
            "data": kwargs.get("data", self.default_task),
            "max_turns": int(kwargs.get("max_turns", self.default_max_turns)),
        }
        # SQLEnv stats the sqlite file in __init__, so build it off the event loop.
        env = await asyncio.to_thread(SQLEnv, Text2SQLEnvConfig(db_path=self.db_root), extras)
        self._remember(instance_id, {"env": env, "reward": 0.0})
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[bool, str, float, dict[str, Any]]:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            # Defensive: normally _handle_pending_state has already created this.
            await self.start_interaction(instance_id, **kwargs)
            entry = self._instance_dict[instance_id]
        self._instance_dict.move_to_end(instance_id)

        env = entry["env"]
        raw_action = _last_assistant_content(messages)
        action, trailing = _truncate_at_stop_tag(raw_action)
        if trailing:
            SQLInteraction._trailing_stop_text += 1
            if SQLInteraction._trailing_stop_text % 1000 == 1:
                # A few characters is normal sglang behaviour. Large or ubiquitous
                # trailing text means the stop strings are not reaching the engine
                # -- i.e. the generation harness is misconfigured.
                logger.warning(
                    "trimmed %d chars after stop tag (occurrence %d); sample=%r",
                    len(trailing),
                    SQLInteraction._trailing_stop_text,
                    trailing[:60],
                )

        # SQLite execution blocks for as long as the query runs. verl drives every
        # concurrent request of this engine from one event loop, so running it
        # inline would serialise the whole rollout batch behind the slowest query
        # -- which would wreck the very throughput measurement this exists for.
        try:
            step_out = await asyncio.to_thread(env.step, action)
        except Exception:
            # One malformed trajectory must not abort a whole training run.
            logger.exception("SQL env step failed for %s; ending trajectory", instance_id)
            self._release(instance_id)
            return True, "", 0.0, {"env_error": True}

        reward = float(step_out["reward"])
        done = bool(step_out["done"])
        observations = step_out["observations"]
        content = observations[0]["content"] if observations else ""

        entry["reward"] = reward
        metadata = dict(step_out.get("metadata") or {})
        metadata["turns"] = env.turns

        # No observation and not done would mean appending an empty user turn and
        # looping; treat it as terminal instead.
        if done or not content:
            self._release(instance_id)
            return True, content, reward, metadata

        return False, content, reward, metadata

    async def calculate_score(self, instance_id: str, **kwargs: Any) -> float:
        entry = self._instance_dict.get(instance_id)
        return float(entry["reward"]) if entry else 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        self._release(instance_id)
