# Caliper: Cost-Bounded Autonomy for Autonomous LLM Agent Fleets

*Draft skeleton for arXiv (cs.AI). Author: Vasundra Srinivasan.*

## Abstract

Autonomous LLM agents are typically deployed with no enforced economic bound: token
usage is metered post hoc and surfaced in traces after a run completes. Under
time-and-materials economics this is tolerable; under fixed-fee or consumption
pricing it is not, because a single agent caught in a reason–act–reason loop can
exhaust the entire economic envelope of a task before a human intervenes. We
present *Caliper*, a runtime that treats the budget as a control-plane primitive:
spend is enforced during execution, not observed after it. Caliper combines (i)
hierarchical cost metering across step/run/session/fleet scopes, (ii) declarative
soft/hard budget policies with graceful degradation, and (iii) online detection of
the structural pathologies that drive overspend — immediate repetition, periodic
cycles, and state oscillation — independent of the agent's goal. We formalize the
detection criteria, characterize their cost and false-positive behavior, and
report an empirical study of budget overrun frequency and the savings from runtime
enforcement on a suite of agentic tasks.

## 1. Introduction

- The economic-control gap in current agent frameworks: observability ≠ control.
- Why pricing-model shifts (T&M → fixed-fee → consumption) make this load-bearing.
- Contributions:
  1. A control-plane formulation of agent budgets along two orthogonal axes — a *temporal* nesting (step/run/session/fleet) and a *dimensional* attribution (per-agent, per-task, per-(agent,task)) — composed and enforced at runtime.
  2. Statistical grounding of cost events against per-scope online baselines (Welford + dual-EWMA), yielding spike and trend signals distinct from, and earlier than, budget exhaustion.
  3. Goal-independent, online detection of the three overspend pathologies (repetition, periodic cycle, state oscillation).
  4. A graceful-degradation policy (soft→downgrade, hard→halt) and its semantics.
  5. An open-source, framework-native reference implementation (LangGraph/LangChain) and a reproducible benchmark.

## 2. Related Work

- Agent frameworks and their cost/observability features (LangSmith, OpenTelemetry-for-LLM, etc.).
- Loop/non-termination handling in agent loops (max-iteration caps and why they are blunt).
- Anytime algorithms and bounded rationality; control theory analogues (circuit breakers, governors).
- Cloud cost-governance and quota systems as prior art for hierarchical budgets.

## 3. Problem Formulation

- Define an agent run as a trajectory of steps; each step has a token/dollar cost and a label vector (agent, task, ...).
- **Two budget axes.** Temporal scopes as a nested containment hierarchy (step ⊂ run ⊂ session ⊂ fleet); dimensional scopes as label projections (per-agent, per-task, per-(agent,task)). A ceiling on any scope, on either axis, bounds the spend attributed to it. The axes are orthogonal and composed: an event updates all scopes it belongs to, and any may breach.
- Why per-dimension matters: a single global ceiling cannot distinguish a normally expensive task from a runaway agent, and absorbs a single bad actor into the aggregate until the whole budget is gone.
- The control objective: halt before crossing a hard ceiling on any scope while minimizing premature halts on healthy runs.

## 3a. Statistical Grounding of Cost Events

- Per-scope online statistics over per-event cost: Welford mean/variance (no stored history) and short/long EWMAs.
- **Spike**: z-score ≥ τ_z, with a relative-deviation fallback when variance is zero (a step change off a flat baseline is undefined under z but is exactly the case of interest).
- **Trend**: short/long EWMA ratio ≥ τ_r (cost-per-event drifting upward for this scope).
- Warmup (min-samples) before judging; per-scope independence so a cheap agent and an expensive agent are each judged against their own norm.
- These are observability signals that fire *before* exhaustion; they do not by themselves halt a run.

## 4. Detection of Overspend Pathologies

- **Action signature**: stable hash of (tool, normalized args); rationale for goal-independence.
- **Immediate repetition**: tail run-length ≥ r.
- **Periodic cycle**: smallest period p whose block recurs c times at the tail.
- **State oscillation**: distinct-signature count ≤ d over a full window of length w (activity without progress).
- Complexity: O(w) per step; memory O(w). Discuss thresholds and the precision/recall tradeoff.

## 5. Enforcement and Graceful Degradation

- Soft ceiling → downgrade (cheaper model, fewer tools, force summarize-and-stop).
- Hard ceiling → halt, raised inside the call stack so the run actually stops.
- Severity ordering (hard > soft; broader scope > narrower at equal severity).

## 6. Implementation

- LangChain callback for provider-agnostic metering; LangGraph conditional-edge gating.
- Thread-safe hierarchical meter; configurable price book with prefix matching.
- LangSmith feedback emission for halted runs (reason codes: budget_exceeded / loop_detected).

## 7. Evaluation

- **Tasks**: a suite of agentic benchmarks (tool-use, web research, multi-step coding) with injected failure conditions.
- **Metrics**: budget-overrun rate without/with Caliper; dollars saved; false-halt rate on healthy runs; detection latency (steps-to-trip).
- **Ablations**: each detector independently; window/threshold sensitivity; soft-vs-hard-only.
- **Baselines**: fixed max-iteration cap; post-hoc cost reporting only.

## 8. Limitations and Future Work

- Signature collisions / adversarial argument churn that defeats signature equality.
- Semantic loops that are not structurally periodic.
- Cross-process fleet accounting (distributed meter); persistence and recovery.
- Learned, adaptive thresholds from telemetry.

## 9. Conclusion

Budgets belong in the control plane. Caliper shows that enforcing them at runtime,
together with goal-independent loop detection, converts unbounded agent autonomy
into a bounded, economically predictable primitive.

## Reproducibility

Code: https://github.com/vasundras/caliper (MIT). Benchmark scripts and seeds in `benchmarks/`.
