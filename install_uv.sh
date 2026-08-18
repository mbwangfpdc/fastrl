#!/usr/bin/env bash
# Reproduce the uv-based FastRL install. See INSTALL-uv.md for the "why".
#
#   ./install_uv.sh          # build patched flashinfer wheel + install everything
#   ./install_uv.sh wheel    # only rebuild the patched flashinfer wheel
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ALWAYS target this repo's own .venv. Do NOT fall back to an inherited
# $VIRTUAL_ENV: shells on this box frequently have another project's venv
# (e.g. granular-cais-rl) already activated, and uv would then install this
# entire stack -- including a torch downgrade -- straight into that env,
# potentially on top of a live training run. Override deliberately with
# FASTRL_VENV=... if you really want a different target.
export VIRTUAL_ENV="${FASTRL_VENV:-$ROOT/.venv}"
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  echo "warning: ignoring UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT" >&2
  unset UV_PROJECT_ENVIRONMENT
fi
echo "==> Target venv: $VIRTUAL_ENV"
# / is ~90% full on this box; keep the (multi-GB) uv cache on /local_nvme1.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/local_nvme1/mborjigi/.cache/uv}"

FI_VER=0.4.0
TVM_FFI_VER=0.1.0        # 0.1.0b15 was deleted from PyPI; 0.1.0 is the same line's final
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

build_flashinfer_wheel() {
  local whl="$ROOT/wheelhouse/flashinfer_python-${FI_VER}-py3-none-any.whl"
  if [[ -f "$whl" ]]; then
    echo "==> flashinfer wheel already present: $whl"
    return
  fi
  echo "==> Building flashinfer ${FI_VER} wheel with apache-tvm-ffi pin -> ${TVM_FFI_VER}"
  local work
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' RETURN

  curl -sSL -o "$work/fi.tar.gz" \
    "https://files.pythonhosted.org/packages/source/f/flashinfer-python/flashinfer_python-${FI_VER}.tar.gz"
  tar xzf "$work/fi.tar.gz" -C "$work"

  local src="$work/flashinfer_python-${FI_VER}"
  # Two places carry the dead pin: the runtime deps and the PEP 517 build deps.
  sed -i "s/apache-tvm-ffi==0\.1\.0b15/apache-tvm-ffi==${TVM_FFI_VER}/" \
    "$src/pyproject.toml" "$src/requirements.txt"

  mkdir -p "$ROOT/wheelhouse"
  uv build --wheel -o "$ROOT/wheelhouse" "$src"
}

install_all() {
  # ONE resolution for all three targets: verl pins numpy<2.0.0 while sglang
  # wants numpy 2.x, and installing them separately leaves scipy built against
  # numpy>=2 on top of numpy 1.26 (breaks at import with np.long).
  echo "==> Installing sglang[all] + verl + flash-attn in a single resolution"
  uv pip install \
    --find-links "$ROOT/wheelhouse" \
    --override   "$ROOT/uv-overrides.txt" \
    -e "$ROOT/third-party/sglang/python[all]" \
    -e "$ROOT" \
    "$FLASH_ATTN_WHL"
}

smoke_test() {
  echo "==> Smoke test"
  "$VIRTUAL_ENV/bin/python" - <<'PY'
import importlib
mods = ['numpy','scipy','torch','transformers','flash_attn','tvm_ffi','flashinfer',
        'sgl_kernel','sglang','ray','tensordict','sentence_transformers','decord','verl']
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, repr(e)[:120]))
from sglang.srt.entrypoints.engine import Engine                      # noqa: F401
import sglang.srt.speculative.eagle_worker                            # noqa: F401
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout  # noqa: F401
import torch
print('import failures:', bad or 'none')
print('cuda:', torch.cuda.is_available(), torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY
}

if [[ ! -d "$VIRTUAL_ENV" ]]; then
  echo "==> Creating venv at $VIRTUAL_ENV"
  uv venv --python 3.12 "$VIRTUAL_ENV"
fi

build_flashinfer_wheel
if [[ "${1:-}" == "wheel" ]]; then
  echo "==> Wheel only; skipping install."
  exit 0
fi
install_all
smoke_test
echo "==> Done. See INSTALL-uv.md for gotchas."
