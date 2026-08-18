"""Per-LLM-request rollout trace capture, for the sglang-vs-vLLM engine A/B.

Records, for every trajectory, the prefill and decode size of every engine call
plus the environment service time between them. That is the workload
description granular-cais-rl's ``[replay]`` consumes (see its ``replay.py``), so
a trace taken here can be replayed byte-for-byte through vLLM and the two
engines compared on an identical workload.

Enabled only by ``FASTRL_ROLLOUT_TRACE=<path>``; unset, every entry point is a
cheap no-op and the rollout is untouched.

Written per process, not per run: rollout runs in one WorkerDict per GPU, all
appending concurrently. Each process owns ``<path>.<pid>.jsonl`` and the
converter merges them, which avoids relying on O_APPEND write atomicity for
records whose size we do not control.

Turn boundaries are *derived*, not asserted. Because every turn resends the full
context, turn ``t``'s observation size falls out of consecutive prefills:

    obs_tokens[t] = prefill[t+1] - (prefill[t] + decode[t])

so the trace only has to store what the engine itself reported
(``meta_info.prompt_tokens`` / ``completion_tokens``) and the converter
reconstructs the rest. ``cached_tokens`` is carried too -- it is not needed to
replay, but it is how we check that vLLM's prefix cache saw the same
opportunities sglang's radix cache did rather than silently doing more prefill.
"""

from __future__ import annotations

import json
import os
import threading

_PATH = os.environ.get("FASTRL_ROLLOUT_TRACE", "").strip()

_lock = threading.Lock()
_fh = None
_step = -1


def enabled() -> bool:
    return bool(_PATH)


def set_step(step: int) -> None:
    """Tag subsequent records with the global RL step (called by the trainer)."""
    global _step
    _step = int(step)


def current_step() -> int:
    return _step


def _handle():
    global _fh
    if _fh is None:
        path = f"{_PATH}.{os.getpid()}.jsonl"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _fh = open(path, "a", buffering=1)
    return _fh


def emit(record: dict) -> None:
    """Append one trajectory record. No-op unless tracing is enabled."""
    if not _PATH:
        return
    line = json.dumps(record)
    with _lock:
        _handle().write(line + "\n")


def close() -> None:
    global _fh
    with _lock:
        if _fh is not None:
            try:
                _fh.close()
            finally:
                _fh = None
