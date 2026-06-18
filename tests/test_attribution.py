"""Tests for dimensional attribution and composable per-scope budgets."""

from caliper.attribution import AttributionBudget, BudgetRule, LabeledMeter
from caliper.budget import Action
from caliper.labels import attribution_keys, canonical


def test_attribution_keys_cover_dimensions_and_combo():
    keys = attribution_keys({"agent": "researcher", "task": "t1"})
    assert (("agent", "researcher"),) in keys      # per-agent rollup
    assert (("task", "t1"),) in keys               # per-task rollup
    assert canonical({"agent": "researcher", "task": "t1"}) in keys  # the exact pair


def test_labeled_meter_rolls_up_per_dimension():
    m = LabeledMeter()
    m.record({"agent": "a1", "task": "t1"}, 0, 0, 0.40)
    m.record({"agent": "a1", "task": "t2"}, 0, 0, 0.30)
    # a1 across both tasks = 0.70; each task isolated.
    assert round(m.usage((("agent", "a1"),)).usd, 2) == 0.70
    assert round(m.usage((("task", "t1"),)).usd, 2) == 0.40
    assert round(m.usage((("task", "t2"),)).usd, 2) == 0.30


def test_per_task_hard_ceiling_independent_of_agent():
    budget = AttributionBudget([BudgetRule(per="task", usd_hard=0.50)])
    m = LabeledMeter()
    touched = m.record({"agent": "a1", "task": "t1"}, 0, 0, 0.60)
    breaches = budget.evaluate(m, touched)
    assert any(b.scope == (("task", "t1"),) and b.action is Action.HALT for b in breaches)


def test_rules_compose_across_dimensions():
    # Agent is fine ($5 hard), but the task ceiling ($0.5) is blown by one event.
    budget = AttributionBudget([
        BudgetRule(per="agent", usd_hard=5.0),
        BudgetRule(per="task", usd_hard=0.5),
    ])
    m = LabeledMeter()
    touched = m.record({"agent": "a1", "task": "t1"}, 0, 0, 0.75)
    breaches = budget.evaluate(m, touched)
    scopes = {b.scope for b in breaches}
    assert (("task", "t1"),) in scopes
    assert (("agent", "a1"),) not in scopes  # agent still within budget


def test_match_targets_one_value():
    # Only the 'expensive' agent has a tight ceiling; others unbounded by this rule.
    budget = AttributionBudget([BudgetRule(per="agent", match="expensive", usd_hard=0.10)])
    m = LabeledMeter()
    t1 = m.record({"agent": "cheap", "task": "t1"}, 0, 0, 1.00)
    assert budget.evaluate(m, t1) == []
    t2 = m.record({"agent": "expensive", "task": "t2"}, 0, 0, 0.20)
    assert any(b.scope == (("agent", "expensive"),) for b in budget.evaluate(m, t2))
