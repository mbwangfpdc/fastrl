#!/usr/bin/env bash
# Build eagle-train's own venv, isolated from fastrl's and granular-cais-rl's.
#
# WHY THIS IS ITS OWN ENVIRONMENT: eagle-train pins torch==2.6.0 /
# transformers==4.51.1, against fastrl's torch 2.8.0 / transformers 4.57.1 and
# granular-cais-rl's torch 2.10.0. Installing these into either of those breaks
# them -- granular's flash-attn and vLLM `_C` are built against one exact
# libtorch ABI, and a downgrade makes every rollout die minutes into startup
# with an `undefined symbol: ...c10_cuda_check_implementation...` that looks
# nothing like a dependency problem. That has already happened once on this box.
#
# THE SPECIFIC TRAP: shells here export VIRTUAL_ENV pointing at
# granular-cais-rl/.venv. `uv pip install` honours an inherited VIRTUAL_ENV, so
# a bare invocation installs into THAT environment -- under a live training run.
# Hence: set VIRTUAL_ENV explicitly (never ${VIRTUAL_ENV:-...}), and always call
# the interpreter by absolute path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIRTUAL_ENV="${EAGLE_VENV:-$ROOT/.venv}"     # NOT ${VIRTUAL_ENV:-...}
PY="$VIRTUAL_ENV/bin/python"

if [[ ! -d "$VIRTUAL_ENV" ]]; then
  echo "==> building venv at $VIRTUAL_ENV"
  uv venv --python 3.10 "$VIRTUAL_ENV"
else
  echo "==> venv already exists at $VIRTUAL_ENV; reusing"
fi

echo "==> installing pinned requirements"
VIRTUAL_ENV="$VIRTUAL_ENV" uv pip install --python "$PY" -r "$ROOT/requirements.txt"

# `uv venv` does not seed setuptools, but triton's nvidia backend imports it at
# module load (triton/runtime/build.py). Without it, `import deepspeed` dies with
# a ModuleNotFoundError surfacing through transformers' lazy OPT import, and
# every deepspeed op reports "compatibility check failed".
VIRTUAL_ENV="$VIRTUAL_ENV" uv pip install --python "$PY" setuptools wheel

# flash-attn wheel must match cu12 + torch2.6 + cp310 exactly. Prebuilt: no
# compile, no nvcc needed.
#
# NB: cxx11abiFALSE, NOT the abiTRUE wheel the upstream README names. PyPI's
# torch 2.6 is built with the pre-C++11 string ABI (`torch._C.
# _GLIBCXX_USE_CXX11_ABI` is False; that only flipped in torch 2.7), so the
# abiTRUE wheel imports with
#   undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_string...
# and takes deepspeed down with it via transformers' lazy OPT import.
FA_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
echo "==> installing flash-attn ($FA_WHL)"
VIRTUAL_ENV="$VIRTUAL_ENV" uv pip install --python "$PY" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FA_WHL}"

echo "==> verifying (and proving we did NOT touch the other venvs)"
"$PY" - <<'PYEOF'
import torch, transformers
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
gpu = torch.cuda.is_available()
print(f"  cuda avail   {gpu} ({torch.cuda.device_count()} devices)")
assert torch.__version__.startswith("2.6"), "torch must be 2.6.x here"
assert transformers.__version__.startswith("4.51"), "transformers must be 4.51.x here"

# deepspeed.ops.transformer.inference.triton decorates a kernel with
# @triton.autotune at import time, which resolves triton's active driver
# backend eagerly -- on a node with no GPU at all (0 registered backends,
# not even a CPU one) that raises before printing anything, so this can only
# be checked on a node that actually has a GPU driver present.
try:
    import deepspeed
    print(f"  deepspeed    {deepspeed.__version__}")
except Exception as e:
    if gpu:
        raise
    print(f"  deepspeed    SKIPPED (no GPU driver on this node) -- {type(e).__name__}: {e}"[:200])
PYEOF

echo "==> eagle-train venv ready: $PY"
