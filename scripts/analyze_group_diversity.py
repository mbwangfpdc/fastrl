#!/usr/bin/env python3
"""Measure within-GRPO-group diversity of rollouts dumped by FASTRL_ROLLOUT_ONLY=1.

The zero-signal-group count that RESUME_TLT_DRAFTER.md reports is an *indirect*
proxy for diversity collapse: a group is dropped when all n_resp rollouts earn
the same reward, which can happen either because the model is uniformly bad
(all wrong) or because the rollouts are literally the same string. Those two
have completely different causes and completely different fixes, and the
zero-signal counter cannot tell them apart.

This reads the pre-filter dump (all n prompts x n_resp rollouts, before
filter_zero_signal_groups removes anything) and separates them:

  exact_dup_rate  fraction of groups whose rollouts are ALL byte-identical
                  -> sampling itself has collapsed (an SD correctness bug)
  mean_uniq       mean distinct completions per group (n_resp = fully diverse)
  score_var_zero  fraction of groups with zero reward variance (what the
                  zero-signal filter actually keys on)

If SD-on shows a high exact_dup_rate and SD-off does not, the collapse is
speculative decoding breaking sampling diversity. If both show diverse text but
identical scores, the model is just uniformly wrong and SD is not the culprit.

  python scripts/analyze_group_diversity.py <dump.jsonl> [--n-resp 5]
"""
import argparse, json, statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n-resp", type=int, default=5)
    a = ap.parse_args()

    rows = []
    with open(a.path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("empty dump")
        return

    # Group by prompt text: the n_resp rollouts of one GRPO group share a prompt.
    groups = defaultdict(list)
    for r in rows:
        key = r.get("input") or r.get("prompt") or ""
        groups[key].append(r)

    full = [g for g in groups.values() if len(g) == a.n_resp]
    print(f"rows={len(rows)}  groups={len(groups)}  full_groups={len(full)} (n_resp={a.n_resp})")
    if not full:
        print("no complete groups; check --n-resp / dump format")
        return

    exact_dup, uniqs, zero_var, lens = 0, [], 0, []
    turns, sql_tags, sol_tags = [], [], []
    for g in full:
        outs = [(r.get("output") or r.get("response") or "") for r in g]
        u = len(set(outs))
        uniqs.append(u)
        if u == 1:
            exact_dup += 1
        lens.extend(len(o) for o in outs)
        for o in outs:
            # Multi-turn structure. Each env reply comes back as an
            # <observation> block, so this counts completed tool round-trips;
            # a trajectory that never terminates a <sql> block never gets one.
            turns.append(o.count("<observation>"))
            sql_tags.append(o.count("</sql>"))
            sol_tags.append(o.count("<solution>"))
        sc = [r.get("score") for r in g if r.get("score") is not None]
        if len(sc) == len(g) and (max(sc) - min(sc)) == 0:
            zero_var += 1

    n = len(full)
    print(f"exact_dup_rate  {exact_dup/n:.3f}  ({exact_dup}/{n} groups all-identical)")
    print(f"mean_uniq       {statistics.mean(uniqs):.2f} / {a.n_resp}")
    print(f"score_var_zero  {zero_var/n:.3f}  ({zero_var}/{n} groups zero reward variance)")
    print(f"resp_chars      mean={statistics.mean(lens):.0f} max={max(lens)}")
    if not any(r.get("score") is not None for r in rows):
        print("(no score field in this dump -- rollout-only mode; diversity numbers above are the point)")

    print(f"turns/traj      mean={statistics.mean(turns):.2f} "
          f"zero_turn_frac={sum(1 for t in turns if t==0)/len(turns):.3f}")
    print(f"closed </sql>   mean={statistics.mean(sql_tags):.2f} "
          f"none_frac={sum(1 for t in sql_tags if t==0)/len(sql_tags):.3f}")
    print(f"<solution>      present_frac={sum(1 for t in sol_tags if t>0)/len(sol_tags):.3f}")

    hist = defaultdict(int)
    for u in uniqs:
        hist[u] += 1
    print("uniq-per-group histogram: " + "  ".join(
        f"{k}:{hist[k]}" for k in sorted(hist)))


if __name__ == "__main__":
    main()
