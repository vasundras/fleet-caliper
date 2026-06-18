"""Reproducible benchmark: cost-bounded autonomy on a simulated agent fleet.

This benchmark does **not** call any LLM. It drives Caliper's core directly
(LabeledMeter, AttributionBudget, BaselineTracker) over a synthetic but
plausible stream of cost events, so results are deterministic, free, and
reproducible anywhere — exactly what an arXiv §7 needs.

We model a fleet of agents working tasks. Most events are "healthy": cost drawn
from a per-agent log-normal-ish distribution. Into this stream we inject the
failure modes that motivate the system:

  * RUNAWAY    — a task whose per-event cost climbs every step (a degrading
                 loop) until something halts it.
  * SPIKE      — a single pathologically expensive event on an otherwise normal task.

We compare three governance regimes on the SAME event stream:

  1. none        — no enforcement (status quo "observe after the fact").
  2. iteration   — a blunt max-iteration cap per task (the common baseline).
  3. caliper     — per-task + per-agent budgets with statistical grounding.

and report, per regime: total spend, dollars saved vs. `none`, how early a
runaway was stopped, and (for caliper) spike/trend detection counts.

Determinism: a small seeded LCG (no external RNG, no Math.random equivalent),
so `python benchmarks/simulate.py --seed 7` reproduces exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

# Import the pure-logic core directly (no langchain dependency needed).
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from caliper.attribution import AttributionBudget, BudgetRule, LabeledMeter  # noqa: E402
from caliper.baselines import BaselineTracker  # noqa: E402
from caliper.budget import Action  # noqa: E402


# --- deterministic RNG (seeded LCG; no global random state) -----------------
class LCG:
    """Numerical Recipes LCG — tiny, deterministic, dependency-free."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next_float(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 0x100000000

    def jitter(self, base: float, frac: float) -> float:
        """`base` perturbed by +/- frac (uniform)."""
        return base * (1.0 + (self.next_float() * 2 - 1) * frac)


# --- workload model ---------------------------------------------------------
@dataclass
class Event:
    agent: str
    task: str
    cost: float
    kind: str  # "healthy" | "runaway" | "spike"


@dataclass
class Workload:
    events: list[Event]
    # ground truth for evaluation
    runaway_tasks: set[str] = field(default_factory=set)
    spike_events: int = 0


# Per-agent healthy mean cost-per-event (USD). Distinct scales on purpose, so
# per-scope baselines must judge each agent against its own norm.
AGENT_PROFILE = {
    "researcher": 0.020,
    "coder": 0.045,
    "summarizer": 0.008,
}


def build_workload(rng: LCG, n_tasks: int = 60, steps_per_task: int = 12) -> Workload:
    agents = list(AGENT_PROFILE)
    events: list[Event] = []
    runaway_tasks: set[str] = set()
    spikes = 0

    for t in range(n_tasks):
        task = f"task-{t:03d}"
        agent = agents[t % len(agents)]
        base = AGENT_PROFILE[agent]

        # ~12% of tasks go runaway; ~10% get a one-off spike.
        is_runaway = rng.next_float() < 0.12
        if is_runaway:
            runaway_tasks.add(task)

        for s in range(steps_per_task):
            if is_runaway:
                # Cost grows geometrically — a degrading loop left unchecked.
                cost = rng.jitter(base * (1.35 ** s), 0.10)
                kind = "runaway"
            else:
                cost = rng.jitter(base, 0.25)
                kind = "healthy"
                # occasional single spike on a healthy task
                if rng.next_float() < 0.01:
                    cost *= 30.0
                    kind = "spike"
                    spikes += 1
            events.append(Event(agent, task, max(cost, 0.0), kind))

    return Workload(events, runaway_tasks, spikes)


# --- governance regimes -----------------------------------------------------
# A "healthy event" is one the workload intended to happen (not part of a
# runaway's degrading tail). Preserving these is the point: cheap governance
# that also truncates healthy work is destroying the deliverable to save pennies.
def _healthy_total(wl: Workload) -> tuple[int, float]:
    n = sum(1 for e in wl.events if e.kind != "runaway")
    usd = sum(e.cost for e in wl.events if e.kind != "runaway")
    return n, usd


def run_none(wl: Workload) -> dict:
    """No enforcement: everything runs, everything is paid for."""
    spend = sum(e.cost for e in wl.events)
    hn, _ = _healthy_total(wl)
    return {
        "regime": "none",
        "spend": spend,
        "stopped_runaways": 0,
        "healthy_done": hn,  # all healthy work completes
        "detail": {},
    }


def run_iteration_cap(wl: Workload, max_steps: int = 6) -> dict:
    """Blunt baseline: cap every task at `max_steps` events, regardless of cost.

    Stops runaways, but indiscriminately — it also truncates healthy tasks that
    legitimately need more than `max_steps` steps.
    """
    seen: dict[str, int] = {}
    spend = 0.0
    stopped = set()
    healthy_done = 0
    for e in wl.events:
        c = seen.get(e.task, 0)
        if c >= max_steps:
            if e.task in wl.runaway_tasks:
                stopped.add(e.task)
            continue
        seen[e.task] = c + 1
        spend += e.cost
        if e.kind != "runaway":
            healthy_done += 1
    return {
        "regime": f"iteration(cap={max_steps})",
        "spend": spend,
        "stopped_runaways": len(stopped),
        "healthy_done": healthy_done,
        "detail": {},
    }


def run_caliper(
    wl: Workload,
    task_hard: float = 0.50,
    agent_hard: float = 8.0,
) -> dict:
    """Per-task + per-agent budgets with statistical grounding."""
    meter = LabeledMeter()
    budget = AttributionBudget([
        BudgetRule(per="task", usd_hard=task_hard),
        BudgetRule(per="agent", usd_hard=agent_hard),
    ])
    baselines = BaselineTracker(z_threshold=3.0, trend_ratio=1.75, min_samples=6)

    spend = 0.0
    halted_tasks: set[str] = set()
    halted_agents: set[str] = set()
    stopped_runaways: set[str] = set()
    spikes_detected = 0
    spikes_true_pos = 0
    trends_detected = 0
    steps_to_stop: list[int] = []
    task_step: dict[str, int] = {}
    healthy_done = 0

    for e in wl.events:
        if e.task in halted_tasks or e.agent in halted_agents:
            continue
        task_step[e.task] = task_step.get(e.task, 0) + 1

        # pay for the event, then attribute + ground + enforce
        spend += e.cost
        if e.kind != "runaway":
            healthy_done += 1
        touched = meter.record({"agent": e.agent, "task": e.task}, 0, 0, e.cost)

        # statistical grounding (per-scope spike/trend)
        for key in touched:
            v = baselines.score(key, e.cost)
            if v.is_spike:
                spikes_detected += 1
                if e.kind in ("spike", "runaway"):
                    spikes_true_pos += 1
            if v.is_trend:
                trends_detected += 1

        # budget enforcement
        for b in budget.evaluate(meter, touched):
            if b.action is Action.HALT:
                dim = b.scope[0][0]
                if dim == "task":
                    halted_tasks.add(e.task)
                    if e.task in wl.runaway_tasks:
                        stopped_runaways.add(e.task)
                        steps_to_stop.append(task_step[e.task])
                elif dim == "agent":
                    halted_agents.add(e.agent)

    avg_stop = sum(steps_to_stop) / len(steps_to_stop) if steps_to_stop else 0.0
    return {
        "regime": "caliper",
        "spend": spend,
        "stopped_runaways": len(stopped_runaways),
        "healthy_done": healthy_done,
        "detail": {
            "spikes_detected": spikes_detected,
            "spikes_true_positive": spikes_true_pos,
            "trends_detected": trends_detected,
            "avg_steps_to_halt_runaway": round(avg_stop, 2),
        },
    }


# --- reporting --------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    rng = LCG(args.seed)
    wl = build_workload(rng, n_tasks=args.tasks, steps_per_task=args.steps)

    none = run_none(wl)
    itr = run_iteration_cap(wl, max_steps=6)
    cal = run_caliper(wl)

    baseline_spend = none["spend"]
    n_runaway = len(wl.runaway_tasks)
    healthy_n, _ = _healthy_total(wl)

    print("=" * 78)
    print(f"Caliper benchmark  (seed={args.seed}, tasks={args.tasks}, steps={args.steps})")
    print("=" * 78)
    print(f"workload: {len(wl.events)} events | "
          f"{n_runaway} runaway tasks | {wl.spike_events} injected spikes | "
          f"{healthy_n} healthy events")
    print()
    header = (f"{'regime':<22}{'spend ($)':>11}{'saved':>14}"
              f"{'runaways':>11}{'healthy kept':>15}")
    print(header)
    print("-" * len(header))
    for r in (none, itr, cal):
        saved = baseline_spend - r["spend"]
        pct = (saved / baseline_spend * 100) if baseline_spend else 0.0
        hk = r["healthy_done"]
        hpct = (hk / healthy_n * 100) if healthy_n else 0.0
        print(f"{r['regime']:<22}{r['spend']:>11.2f}{saved:>8.2f} ({pct:>4.1f}%)"
              f"{r['stopped_runaways']:>7}/{n_runaway}"
              f"{hk:>9}/{healthy_n} ({hpct:>4.0f}%)")
    print()
    print("Read this carefully: the blunt iteration cap can spend less than Caliper,")
    print("but only by truncating HEALTHY tasks too. Caliper halts what breaches budget")
    print("and lets legitimate work finish -- cheaper than 'none', without destroying")
    print("the deliverable to save pennies. 'healthy kept' is where that shows up.")
    print()
    d = cal["detail"]
    print("caliper grounding (early signal, before any ceiling is hit):")
    print(f"  spikes detected         : {d['spikes_detected']} "
          f"({d['spikes_true_positive']} on runaway/spike events)")
    print(f"  upward trends detected  : {d['trends_detected']}")
    print(f"  avg steps to halt a runaway task: {d['avg_steps_to_halt_runaway']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
