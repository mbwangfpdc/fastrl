#!/usr/bin/env python
"""Shrink an exported EAGLE drafter for transfer, and rebuild it on the far side.

An exported drafter is 1.61 GB, but only 517 MB of that was ever trained:
`embed_tokens.weight` (1.09 GB) is a verbatim, frozen copy of the TARGET model's
embedding, which any node can reproduce from the target checkpoint. Moving it
between machines is pure waste -- and it is what pushes the directory past what
a git host will take.

    # on the old node
    python slim_drafter.py strip --drafter <dir> --out <slim_dir>

    # on the new node (target comes from the HF cache / hub)
    python slim_drafter.py rehydrate --slim <slim_dir> \
        --target Qwen/Qwen2.5-Coder-7B-Instruct --out <drafter_dir>

`rehydrate` reproduces a byte-identical drafter directory, so sglang's
`speculative_draft_model_path` sees exactly what `export_drafter.py` produced.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

EMBED_KEY = "embed_tokens.weight"


def _require_drafter_dir(path: Path, flag: str) -> Path:
    """Resolve and check a drafter directory, reporting what was actually looked at.

    A bare relative path is the easy mistake here (the docs used to show one),
    and torch.load's FileNotFoundError does not say which directory it resolved
    against -- so say it.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(
            f"{flag} {path} does not exist (resolved to {resolved}).\n"
            f"Pass an ABSOLUTE path, or cd to the directory holding it. If you are "
            f"on a fresh node, the weights are not in git -- copy the drafter over "
            f"first (rsync/scp), then rerun."
        )
    if not (resolved / "pytorch_model.bin").is_file():
        have = ", ".join(sorted(p.name for p in resolved.iterdir())) or "(empty)"
        raise SystemExit(
            f"{flag} {resolved} has no pytorch_model.bin. Contains: {have}"
        )
    return resolved


def _copy_sidecars(src: Path, dst: Path) -> None:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        f = src / name
        if f.exists():
            shutil.copy2(f, dst / name)


def strip(args) -> int:
    src, out = _require_drafter_dir(Path(args.drafter), "--drafter"), Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sd = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if EMBED_KEY not in sd:
        print(f"{EMBED_KEY} already absent; copying as-is")
    else:
        del sd[EMBED_KEY]
    torch.save(sd, out / "pytorch_model.bin")
    _copy_sidecars(src, out)
    before = (src / "pytorch_model.bin").stat().st_size
    after = (out / "pytorch_model.bin").stat().st_size
    print(f"{before/1e9:.2f} GB -> {after/1e6:.0f} MB  ({len(sd)} tensors) -> {out}")
    return 0


def rehydrate(args) -> int:
    slim, out = _require_drafter_dir(Path(args.slim), "--slim"), Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sd = torch.load(slim / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if EMBED_KEY in sd:
        print(f"{EMBED_KEY} already present; nothing to rehydrate")
    else:
        from transformers import AutoModelForCausalLM

        print(f"loading target {args.target} to recover {EMBED_KEY} ...")
        tgt = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16)
        sd[EMBED_KEY] = tgt.model.embed_tokens.weight.detach().clone()
        del tgt
    torch.save(sd, out / "pytorch_model.bin")
    _copy_sidecars(slim, out)
    print(f"wrote {out} ({len(sd)} tensors, "
          f"{(out / 'pytorch_model.bin').stat().st_size/1e9:.2f} GB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strip", help="drop the target-derived embedding")
    s.add_argument("--drafter", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=strip)
    r = sub.add_parser("rehydrate", help="re-attach the embedding from the target")
    r.add_argument("--slim", required=True)
    r.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    r.add_argument("--out", required=True)
    r.set_defaults(func=rehydrate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
