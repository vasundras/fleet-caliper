# Caliper

**Cost-bounded autonomy for LLM agent fleets.** A small, framework-native runtime that gives autonomous agents a hard economic ceiling and a way to detect — and stop — the failure modes that quietly burn budget: runaway loops, tool thrashing, and oscillating state.

Built for [LangGraph](https://github.com/langchain-ai/langgraph) / [LangChain](https://github.com/langchain-ai/langchain), with first-class [LangSmith](https://smith.langchain.com/) telemetry.

> A caliper measures and bounds with precision. So does this. Autonomy without a cost bound is not autonomy — it is an unpriced liability. A single agent stuck in a reason→act→reason loop can consume an entire budget before a human notices. Caliper makes the economic envelope a first-class, enforced runtime constraint rather than a dashboard you read after the money is gone.

## Why this exists

Most agent frameworks meter cost *after the fact*: you read token usage in a trace once the run is done. That is fine for analytics and useless for control. When delivery economics move from time-and-materials to fixed-fee or consumption pricing, the cost of an agent run stops being an accounting detail and becomes the thing that determines whether the work is profitable at all.

Caliper treats the budget as a **control-plane primitive**:

- **Budgets are enforced at runtime**, not observed after it. A run that exceeds its hard ceiling is *halted*, not logged.
- **Budgets bound two independent axes.** *Temporal*: per-step, per-run, per-session, per-fleet. *Dimensional*: per-agent, per-task, and per-(agent, task) — so a single runaway task or a single expensive agent is bounded on its own, not just the global total. The two axes compose: an event can be within its agent budget yet trip its task budget.
- **Every cost event is grounded statistically** against a learned baseline *for that agent and that task* (online mean/variance + EWMA). You get **spike** alerts (this event is anomalous for this scope) and **trend** alerts (this scope's cost-per-event is drifting up) — distinct from, and earlier than, budget exhaustion.
- **The pathologies that cause overspend are detected directly** — loops, tool thrashing, state oscillation — not inferred from a cost spike after the fact.
- **Degradation is graceful**: trip a soft limit and downgrade (smaller model, fewer tools, force a summary-and-stop) before tripping the hard limit and halting.

## Core ideas

| Primitive | What it does |
|---|---|
| `CostMeter` | Thread-safe, hierarchical accumulation of tokens and dollar cost across step / run / session / fleet scopes. |
| `BudgetPolicy` | Declarative soft and hard ceilings per scope, and the action to take at each (`warn`, `downgrade`, `halt`). |
| `PriceBook` | Configurable token→dollar mapping. Ships with illustrative defaults; you set your own rates. |
| `LabeledMeter` | Dimensional accumulation keyed by labels (agent / task / their combination). The attribution axis. |
| `BudgetRule` / `AttributionBudget` | Composable per-dimension ceilings: "every task gets $0.50", "the `researcher` agent gets $5", evaluated together. |
| `BaselineTracker` | Online per-scope statistics (Welford mean/variance + short/long EWMA) that score each event for **spike** and **trend**. |
| `Alert` / `AlertKind` | `EXHAUSTION` / `SPIKE` / `TREND` alerts routed to your sink (log, LangSmith, pager). |
| `LoopDetector` | Online detection of immediate repetition, periodic cycles, and state oscillation over a sliding window of action signatures. |
| `CaliperCallbackHandler` | A LangChain callback that meters every LLM call, attributes it per label, grounds it against baselines, and trips the breaker the instant a hard ceiling is crossed. |
| `Caliper` | The facade that ties them together and exposes the LangGraph integration points. |

## Install

```bash
pip install caliper-ai   # distribution name; imports as `caliper`
```

## Quickstart (LangGraph)

```python
from caliper import Caliper, BudgetPolicy, PriceBook

caliper = Caliper(
    policy=BudgetPolicy(
        run_usd_soft=0.50,   # downgrade past here
        run_usd_hard=1.00,   # halt past here
        fleet_usd_hard=50.0, # total ceiling across all runs
    ),
    pricebook=PriceBook.default(),
)

# 1. Meter every model call — framework-agnostic, works anywhere callbacks do.
llm = ChatModel(...).with_config(callbacks=[caliper.callback_handler()])

# 2. Gate the graph — route to END the moment the breaker trips.
graph.add_conditional_edges("agent", caliper.budget_edge, {
    "continue": "tools",
    "halt": END,
})

# 3. Feed the loop detector from inside a node.
def agent_node(state):
    caliper.record_step(state)        # raises CaliperTripped on a detected loop
    ...
```

## Per-agent and per-task budgets, with statistical grounding

```python
from caliper import Caliper, BudgetPolicy, AttributionBudget, BudgetRule, BaselineTracker

caliper = Caliper(
    policy=BudgetPolicy(fleet_usd_hard=50.0),         # global backstop
    attribution_budget=AttributionBudget([
        BudgetRule(per="agent", usd_hard=2.00),       # every agent: $2 hard
        BudgetRule(per="task",  usd_soft=0.40,        # every task: warn at $0.40,
                                usd_hard=0.50),        #             halt at $0.50
        BudgetRule(per="agent", match="researcher",   # this one agent gets more room
                                usd_hard=5.00),
    ]),
    baselines=BaselineTracker(z_threshold=3.0, trend_ratio=1.75),
    on_alert=lambda a: print(a),                      # exhaustion / spike / trend
)

# Attach labels per call so spend is attributable to an agent and a task.
llm = model.with_config(
    callbacks=[caliper.callback_handler()],
    metadata={"agent": "researcher", "task": task_id},
)

# Spend is now bounded per agent AND per task AND globally, simultaneously;
# anomalous events page you via on_alert before any ceiling is even reached.
print(caliper.snapshot()["attributed"])   # {'agent=researcher': {...}, 'task=42': {...}, ...}
```

## What's here

```
src/caliper/
  budget.py            # CostMeter, BudgetPolicy, scopes, exceptions
  pricing.py           # PriceBook
  loop_detection.py    # LoopDetector: repetition / cycle / oscillation
  callbacks.py         # CaliperCallbackHandler (LangChain)
  caliper.py           # Caliper facade + budget_edge + record_step
examples/
  langgraph_research_agent.py
tests/
  test_budget.py
  test_loop_detection.py
PAPER.md               # arXiv (cs.AI) write-up skeleton
```

## Status

Early. The interfaces are stable enough to build on; the benchmark and empirical study described in `PAPER.md` are in progress.

## License

MIT © 2026 Vasundra Srinivasan
