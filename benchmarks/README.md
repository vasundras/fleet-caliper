# Caliper benchmarks

Reproducible, LLM-free simulations that drive Caliper's core over a synthetic
agent-fleet workload. No API keys, no network, deterministic by seed.

```bash
python benchmarks/simulate.py --seed 7
python benchmarks/simulate.py --seed 7 --tasks 200 --steps 20
```

## What it models

A fleet of agents (`researcher`, `coder`, `summarizer`, each with its own cost
scale) works a stream of tasks. Most events are healthy; a fraction of tasks go
**runaway** (per-event cost climbs geometrically — a degrading loop), and a few
healthy tasks emit a one-off **spike**.

The same event stream is run under three governance regimes:

| Regime | What it does |
|---|---|
| `none` | No enforcement. Everything runs and is paid for. The status-quo "observe after the fact" baseline. |
| `iteration(cap=N)` | The common blunt baseline: cap every task at N steps regardless of cost. |
| `caliper` | Per-task + per-agent budgets with statistical grounding. |

## The metric that matters: *healthy work kept*

A naive reading says the iteration cap "wins" because it spends the least. It
doesn't — it spends less only by **truncating healthy tasks too**. The benchmark
reports `healthy kept` precisely to expose this:

```
regime                  spend ($)         saved   runaways   healthy kept
none                        34.11    0.00 ( 0.0%)      0/6      648/648 (100%)
iteration(cap=6)            12.62   21.50 (63.0%)      6/6      324/648 ( 50%)
caliper                     18.71   15.40 (45.2%)      6/6      531/648 ( 82%)
```

(seed=7.) The iteration cap saves the most dollars but **destroys half the
legitimate deliverable** to do it. Caliper saves 45% versus unbounded spend,
stops every runaway, and preserves the large majority of healthy work — because
it halts what *breaches budget*, not what *runs long*. Saving money by deleting
good work is not governance; it is amputation.

Caliper's statistical grounding also surfaces runaways and spikes as **alerts**
well before any hard ceiling is hit, which is the signal an operator actually
wants (intervene early) rather than a post-hoc invoice.

## Notes

- Determinism uses a small seeded LCG; `Date.now()`/RNG global state are avoided
  so a given `--seed` reproduces exactly.
- This benchmark exercises the budgeting/attribution/baseline core directly. An
  end-to-end LangGraph benchmark (real models, real tools, injected loops) is
  the natural next artifact and feeds the same table with live numbers.
