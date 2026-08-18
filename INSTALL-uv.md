# Installing FastRL with uv (instead of conda)

The upstream README installs via conda + pip. This repo is installed here with
`uv` into a local `.venv` (Python 3.12.3). This file records the one upstream
breakage you have to work around and the exact commands used.

## The blocker: a deleted PyPI prerelease

The README's `pip install -e "python[all]"` step fails today — **not because of
uv**. The chain is:

    sglang 0.5.3.post2  ->  flashinfer-python==0.4.0  ->  apache-tvm-ffi==0.1.0b15

Every `apache-tvm-ffi` `0.1.0b*` prerelease has been **deleted from PyPI**
(only finals `0.1.0` .. `0.1.13.post3` remain). So that `==0.1.0b15` pin is
unsatisfiable by *any* installer — plain pip and conda hit it too. uv just
reports it more explicitly:

    Because there is no version of apache-tvm-ffi==0.1.0b15 and
    flashinfer-python==0.4.0 depends on apache-tvm-ffi==0.1.0b15, ...

`flashinfer-python` 0.4.0 repeats the pin in **two** places, which is why one
fix is not enough:

1. its runtime `requires-dist`  -> handled by `uv-overrides.txt`
2. its `[build-system].requires` -> overrides do **not** apply to build
   requirements, and 0.4.0 ships as an sdist only (no wheel on PyPI, and no
   wheel on flashinfer.ai for torch2.8), so it must be built from source.

### Why `apache-tvm-ffi==0.1.0` is the right substitute

* `0.1.0b15` -> `0.1.0` is the same release line: b15 is from 2025-10-09,
  the `0.1.0` final from 2025-10-21.
* flashinfer itself relaxed the pin to `apache-tvm-ffi>=0.1,<0.2` in 0.5.0,
  i.e. upstream considers 0.1.0 final compatible.
* flashinfer 0.4.0's `build_backend.py` never imports `tvm_ffi` at all — it is
  pure setuptools packaging, so the build-time pin is vestigial.

Verified after install: `tvm_ffi`, `flashinfer`, `sgl_kernel` and `sglang` all
import cleanly, and `sglang.srt.speculative.eagle_worker` (the module FastRL's
adaptive SD relies on) imports.

## Commands used

```bash
cd /local_nvme1/mborjigi/fastrl
export VIRTUAL_ENV=$PWD/.venv
export UV_CACHE_DIR=/local_nvme1/mborjigi/.cache/uv   # / is ~90% full; keep cache off it

# 1. Build a flashinfer 0.4.0 wheel with the dead pin retargeted to 0.1.0
./install_uv.sh            # see that script; produces wheelhouse/flashinfer_python-0.4.0-py3-none-any.whl

# 2. Resolve sglang + verl + flash-attn TOGETHER (see "one resolution" below)
uv pip install \
  --find-links $PWD/wheelhouse \
  --override   $PWD/uv-overrides.txt \
  -e "third-party/sglang/python[all]" \
  -e . \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

### Install everything in ONE resolution

verl's `setup.py` pins `numpy<2.0.0` while sglang pulls numpy 2.x. If you
install sglang and verl in *separate* `uv pip install` calls, the second call
downgrades numpy to 1.26.4 but leaves `scipy` at a build that requires
numpy>=2 — which imports but then dies with
`AttributeError: module 'numpy' has no attribute 'long'` (this breaks
`sentence_transformers`, among others).

Passing all three targets to a single `uv pip install` lets uv pick a
consistent set (it selects `scipy 1.17.1`, which works with numpy 1.26.4).

## Result

221 packages, `.venv` ~11G. Key versions:

| package | version |
|---|---|
| torch | 2.8.0+cu128 |
| sglang | 0.5.3.post2 (editable, `third-party/sglang/python`) |
| verl | 0.5.0.dev0 (editable, repo root) |
| flashinfer-python | 0.4.0 (local patched wheel) |
| apache-tvm-ffi | 0.1.0 (overridden from 0.1.0b15) |
| sgl-kernel | 0.3.15 |
| flash-attn | 2.8.3+cu12torch2.8cxx11abiTRUE |
| transformers | 4.57.1 |
| numpy | 1.26.4 (verl pin) |
| scipy | 1.17.1 |

## Gotchas for later

* **Beware an inherited `$VIRTUAL_ENV`.** Shells on this box often have
  `granular-cais-rl/.venv` already activated. `uv pip install` targets
  `$VIRTUAL_ENV` when set, so an unqualified install here will land in that
  *other* project's venv — which downgrades its torch 2.10 -> 2.8 and can hit a
  live training run. `install_uv.sh` now pins the target venv explicitly; for
  manual `uv pip` calls always pass `VIRTUAL_ENV=$PWD/.venv` yourself.
  Recovery for the other venv is `uv sync --inexact --frozen` in its repo
  (`--inexact` keeps the ~76 packages it has outside its lock, e.g. the
  OSWorld/playwright/docker deps).

* **Never run `uv pip install -e ".[sglang]"`.** verl's `sglang` extra pins
  `sglang==0.4.9.post6` + `torch==2.7.1`, which would clobber the vendored,
  FastRL-modified sglang 0.5.3.post2 in `third-party/`. The base `-e .` is
  correct.
* **`examples/bench_sd.sh` requests `--attention_backend fa3`.** FA3 needs
  Hopper (sm90). The GPUs on this box are L40S = sm_89 (Ada), so fa3 will not
  run — switch that flag to `fa2`/`flashinfer`/`triton` here.
* Re-running either editable install alone can re-break the numpy/scipy pair;
  prefer re-running the single combined command above.
* `uv-overrides.txt` must be passed on every `uv pip install` into this venv,
  or resolution hits the dead `0.1.0b15` pin again.
