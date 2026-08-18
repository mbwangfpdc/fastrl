#!/usr/bin/env python
"""Convert a DeepSpeed ZeRO drafter checkpoint into a drafter directory sglang can load.

eagle_trainer.py saves ZeRO shards (bf16_zero_pp_rank_*), which nothing at
inference time understands. sglang's `speculative_draft_model_path` wants the
same layout mit-han-lab publishes: a directory holding config.json plus a flat
pytorch_model.bin whose keys carry NO `model.` prefix.

Two transforms matter:
  * strip the leading `model.` from every key -- the reverse of the remap the
    warm start applies on the way in.
  * drop `lm_head.weight`. It is copied from the target and frozen during
    training, and mit-han-lab's reference checkpoint omits it too (13 tensors);
    shipping it would just bloat the file with a duplicate of the target's head.

    python scripts/export_drafter.py \
        --ckpt /local_nvme1/mborjigi/eagle-ckpt/sql-coder7b-warm \
        --tag  global_step2793 \
        --target models/qwen2.5-coder-7b-instruct-target \
        --out   /local_nvme1/mborjigi/eagle-drafters/sql-warm-e3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="DeepSpeed output_dir")
    ap.add_argument("--tag", default=None, help="global_stepN (default: contents of `latest`)")
    ap.add_argument("--target", required=True, help="target model dir, for config derivation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    tag = args.tag or (ckpt / "latest").read_text().strip()
    sys.path.insert(0, str(ckpt))
    from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    print(f"consolidating {ckpt}/{tag} ...")
    state = get_fp32_state_dict_from_zero_checkpoint(str(ckpt), tag=tag)

    out_state, dropped = {}, []
    for k, v in state.items():
        if k == "lm_head.weight" or k.endswith(".lm_head.weight"):
            dropped.append(k)
            continue
        out_state[k[len("model."):] if k.startswith("model.") else k] = v.to(torch.bfloat16)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, out / "pytorch_model.bin")

    # Drafter config: the target's, with a single layer and the Eagle class.
    cfg = json.loads((Path(args.target) / "config.json").read_text())
    cfg["architectures"] = ["Qwen2ForCausalLMEagle"]
    cfg["num_hidden_layers"] = 1
    cfg["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = Path(args.target) / fname
        if src.exists():
            shutil.copy(src, out / fname)

    print(f"wrote {len(out_state)} tensors to {out}/pytorch_model.bin (dropped {dropped})")
    for k in list(out_state)[:3]:
        print(f"   {k}: {tuple(out_state[k].shape)}")


if __name__ == "__main__":
    main()
