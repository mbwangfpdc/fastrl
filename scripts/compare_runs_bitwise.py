#!/usr/bin/env python3
"""Compare two rollout dumps prompt-by-prompt and report exact-match rate.

Intended for the greedy (temperature 0) SD on/off pair. Correct speculative
decoding is distribution-preserving in general, which is awkward to test
directly because two sampled runs legitimately differ token-for-token (they
consume different RNG streams). Under GREEDY decoding that ambiguity is gone:
a draft token is accepted iff it equals the target's argmax, so speculation is
a pure latency optimisation and the emitted text must be BIT-IDENTICAL to a
non-speculative run on the same prompt. Any mismatch here is therefore a
definite correctness bug, with no distributional hand-waving required.

Reports the exact-match rate, where the first divergence occurs, and the length
skew -- if SD-on text diverges and then runs much longer, the defect is
compounding rather than a one-off token slip.

  python scripts/compare_runs_bitwise.py <sdoff.jsonl> <sdon.jsonl>
"""
import argparse, json, statistics
from collections import defaultdict


def load(path):
    """Map prompt -> first response (greedy makes the n_resp copies identical)."""
    out = {}
    dupes = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p = r.get("prompt") or r.get("input") or ""
            t = r.get("response") or r.get("output") or ""
            dupes[p].append(t)
            out.setdefault(p, t)
    # how deterministic was each run internally?
    within = [len(set(v)) for v in dupes.values() if len(v) > 1]
    return out, within


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    a = ap.parse_args()

    A, within_a = load(a.baseline)
    B, within_b = load(a.candidate)
    shared = sorted(set(A) & set(B))
    print(f"baseline={len(A)} prompts  candidate={len(B)} prompts  shared={len(shared)}")
    if within_a:
        print(f"within-group unique (baseline):  mean {statistics.mean(within_a):.2f}")
    if within_b:
        print(f"within-group unique (candidate): mean {statistics.mean(within_b):.2f}")
    if not shared:
        print("no shared prompts -- cannot compare")
        return

    exact = 0
    divs, la, lb = [], [], []
    examples = []
    for p in shared:
        x, y = A[p], B[p]
        la.append(len(x))
        lb.append(len(y))
        if x == y:
            exact += 1
        else:
            d = first_divergence(x, y)
            divs.append(d)
            if len(examples) < 3:
                examples.append((d, x, y))

    n = len(shared)
    print(f"\nEXACT MATCH: {exact}/{n}  ({exact/n:.3f})")
    print(f"resp_chars   baseline mean={statistics.mean(la):.0f}  candidate mean={statistics.mean(lb):.0f}")
    if divs:
        print(f"first divergence char index: median={statistics.median(divs):.0f} "
              f"min={min(divs)} max={max(divs)}")
        for k, (d, x, y) in enumerate(examples):
            print(f"\n--- example {k+1}: diverges at char {d} ---")
            print(f"  baseline  ...{x[max(0,d-90):d]}>>>{x[d:d+90]}")
            print(f"  candidate ...{y[max(0,d-90):d]}>>>{y[d:d+90]}")


if __name__ == "__main__":
    main()
