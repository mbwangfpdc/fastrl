# Choosing a training config for the 35-step SkyRL-SQL e2e run

Config-only tuning (no FastRL implementation changes). 32x5, 2 steps, 4x L40S,
SD off. `update_actor` normalised per token, since generated length -- and hence
the training token count -- varies run to run.

## Result: SP2 + forward_prefetch, budget 4096

| config | ms/token | raw update_actor | reps |
|---|---|---|---|
| **SP2, budget 4096, forward_prefetch=True** | **0.208 / 0.203** | 88.0 / 86.6 s | 2 |
| SP2, budget 8192 | 0.233 / 0.222 | 99.1 / 95.5 s | 2 |
| SP2, budget 8192, forward_prefetch=True | 0.241 / 0.239 | 102.0 / 101.6 s | 2 |
| SP2, budget 4096 (baseline defaults) | 0.249 | 105.1 s | 1 |
| SP1, budget 8192 | 0.304 | 127.3 s | 1 |
| reshard_after_forward=False | 0.322 | 136.7 s | 1 |
| SP4, budget 2048 | 0.328 | 141.8 s | 1 |
| gradient checkpointing off | OOM | | |

**~18% off `update_actor`, replicated.**

## What the numbers overturned

* **SP2 is not the slow part.** It beats SP1 by 17% and SP4 by 26% on the
  training step. The suspicion that Ulysses was costing us was wrong.
* **The knobs are anti-additive.** forward_prefetch alone (0.205 mean) beats
  forward_prefetch + budget 8192 (0.240). A larger microbatch appears to eat the
  prefetch overlap, so tuning them independently would have picked the wrong pair.
* **`reshard_after_forward=False` hurts.** It trades memory for re-gather savings
  that do not materialise here.

## Measure per token, not per step

`gen` varied 40.6-104.2s across configs that cannot affect generation -- a 2.5x
spread on an invariant. Raw `update_actor` seconds inherit that noise through the
token count. Per-token normalisation is what makes these comparable, and the
repeats are what separated a real 18% win from the ~15% noise band.

## Offload is not optional here, but the reason is still open

Every attempt to keep the optimizer resident failed:
* at `mem_fraction_static` 0.30/0.20: sglang `init_memory_pool` raises
  "Not enough memory". **These probes were mis-designed** -- sglang sizes
  `KV = available_after_weights - total*(1 - frac)`, so a LOWER fraction means a
  LARGER reserve and a SMALLER pool. The failures are consistent with the
  fraction alone and say nothing about the optimizer.
* at the known-good 0.6, params still offloaded: no crash, no OOM, no stall --
  it simply never finished even generation in 25 minutes (0 steps). Killed on
  the deadline.

So residency is empirically unusable, but "sglang and the optimizer are
co-resident" is NOT established: verl colocates by timesharing, params were on
CPU throughout, and AdamW allocates its moments lazily on the first step. What
actually consumes the memory at 0.6 is unexplained.

## Harness

`cfg_probe_lib.sh` kills a probe the moment its outcome is decided -- fatal
marker, completion, no log growth for 240s, or an absolute deadline -- and reaps
the engine subprocesses, which outlive the driver and would hold the cards
against the next probe. Before it, a dead actor left the Ray driver waiting and
each bad config burned its full 40-minute timeout; four did. Failures now cost
90-230s.
