#!/usr/bin/env python3
"""Independent Python re-implementation of the `verify_tree_greedy` CUDA
kernel's accept-walk, diffed against dumps of its own inputs and outputs.

Context: RESUME_TLT_DRAFTER.md's `### UPDATE 2026-08-31b` found that a true
batch=1 isolation test (one prompt per rollout engine, zero concurrent
requests) still only reaches 2/4 bitwise match between SD-on and SD-off under
greedy decoding -- ruling out cross-request batch-position noise as the
explanation, and running almost entirely through FastRL's unmodified,
never-patched mainline SD decode path. This narrows it further: is the
compiled `verify_tree_greedy` kernel making the wrong accept decision given
its own inputs (a kernel-logic bug, independent of anything upstream of it),
or is it walking its inputs correctly and the inputs themselves
(`target_predict`, i.e. the target model's own logits at each candidate
position) already differ from what a plain non-speculative decode would
compute at the identical context (context corruption or genuine
floating-point drift in the verify-step's multi-candidate forward pass)?

The kernel's semantics were reverse-engineered from
`third-party/sglang/sgl-kernel/tests/speculative/test_eagle_utils.py`'s
worked example (traced by hand, confirmed digit-for-digit against that test's
expected `predicts`/`accept_index`/`accept_token_num`): for each row, starting
at local tree-node 0 (root), repeatedly (a) write
`predicts[retrive_index[row, cur]] = target_predict[row, cur]`, (b) walk the
node's children via `retrive_next_token` (first child) then
`retrive_next_sibling` (next child) looking for one whose `candidates` value
equals `target_predict[row, cur]`, (c) if found, that child becomes the new
`cur` and its global index is appended to the accepted path; if not found,
stop. `accept_token_num` is the accepted-path length minus 1 (the root is not
itself an acceptance).

Requires dumps produced by eagle_info.py's `EagleVerifyInput.verify` with
`FASTRL_DEBUG_VERIFY_DIR` set (see that function's greedy branch) -- one
`pid<pid>_round<N>.pt` file per verify call (pid-qualified since several
independent rollout engines, one per GPU, write into the same directory),
each holding candidates / retrive_index / retrive_next_token /
retrive_next_sibling / target_predict (the kernel's inputs) plus predicts /
accept_index / accept_length (the kernel's outputs).

  .venv/bin/python scripts/check_verify_tree_greedy.py <dump_dir>
"""
import argparse
import glob
import os

import torch


def python_verify_tree_greedy(candidates, retrive_index, retrive_next_token,
                               retrive_next_sibling, target_predict):
    """Pure-Python re-implementation. Returns (predicts, accept_index_rows,
    accept_token_num) in the same shapes/padding the kernel uses."""
    bs, draft_token_num = candidates.shape
    total = int(retrive_index.max().item()) + 1
    predicts = [-1] * total
    accept_index_rows = []
    accept_token_num = []
    for row in range(bs):
        cur = 0
        path = [int(retrive_index[row, 0])]
        while True:
            global_idx = int(retrive_index[row, cur])
            pred = int(target_predict[row, cur])
            predicts[global_idx] = pred
            child = int(retrive_next_token[row, cur])
            matched = -1
            while child != -1:
                if int(candidates[row, child]) == pred:
                    matched = child
                    break
                child = int(retrive_next_sibling[row, child])
            if matched == -1:
                break
            path.append(int(retrive_index[row, matched]))
            cur = matched
        accept_index_rows.append(path)
        accept_token_num.append(len(path) - 1)
    return predicts, accept_index_rows, accept_token_num


def check_one(path, verbose):
    d = torch.load(path, map_location="cpu")
    candidates = d["candidates"]
    retrive_index = d["retrive_index"]
    retrive_next_token = d["retrive_next_token"]
    retrive_next_sibling = d["retrive_next_sibling"]
    target_predict = d["target_predict"]

    py_predicts, py_accept_rows, py_accept_len = python_verify_tree_greedy(
        candidates, retrive_index, retrive_next_token, retrive_next_sibling,
        target_predict,
    )

    kern_predicts = d["predicts"].tolist()
    kern_accept_index = d["accept_index"].tolist()
    kern_accept_len = d["accept_length"].tolist()

    ok = True
    # `predicts` is `torch.empty(...)` in production (uninitialized scratch),
    # vs `torch.full(..., -1)` in the kernel's own unit test -- so positions
    # NEITHER side ever visited legitimately hold GPU garbage in the real
    # dump, not -1. Only compare positions our own re-walk actually visited;
    # a mismatch there is where an ACTUAL bug in either predicts or the walk
    # would show up.
    visited = sorted({idx for row in py_accept_rows for idx in row})
    predicts_mismatch = [(i, py_predicts[i], kern_predicts[i]) for i in visited
                         if py_predicts[i] != kern_predicts[i]]
    if predicts_mismatch:
        ok = False
        if verbose:
            print(f"  predicts MISMATCH at visited positions: {predicts_mismatch}")
    if py_accept_len != kern_accept_len:
        ok = False
        if verbose:
            print(f"  accept_length MISMATCH python={py_accept_len} kernel={kern_accept_len}")
    for row, (py_path, kern_row) in enumerate(zip(py_accept_rows, kern_accept_index)):
        kern_path = [x for x in kern_row if x != -1]
        if py_path != kern_path:
            ok = False
            if verbose:
                print(f"  row{row} accept_index MISMATCH python={py_path} kernel={kern_path}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("-v", "--verbose", action="store_true",
                     help="print the actual mismatching tensors, not just counts")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dump_dir, "*round*.pt")))
    if not paths:
        print(f"no round*.pt files found in {a.dump_dir}")
        return

    n_ok = 0
    n_bad = 0
    for p in paths:
        if check_one(p, a.verbose):
            n_ok += 1
        else:
            n_bad += 1
            print(f"{os.path.basename(p)}: KERNEL DISAGREES WITH ITS OWN INPUTS")

    print(f"\n{n_ok}/{len(paths)} verify rounds: kernel's accept decision matches "
          f"an independent Python re-walk of its own inputs.")
    if n_bad:
        print(f"{n_bad} rounds disagree -- that is a genuine kernel-logic bug, "
              f"independent of anything upstream of verify_tree_greedy.")
    else:
        print("Kernel logic is clean on everything it was asked to decide here. "
              "The residual SD/non-SD divergence must therefore trace to "
              "target_predict itself -- i.e. the target model's logits at each "
              "candidate position -- differing from what a plain "
              "non-speculative decode would compute at the identical context "
              "(context corruption, of the same family as the bug already "
              "fixed, or genuine floating-point drift in the verify step's "
              "multi-candidate forward-pass shape).")


if __name__ == "__main__":
    main()
