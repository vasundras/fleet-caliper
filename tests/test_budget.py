"""Tests for cost metering and budget policy enforcement."""

from caliper.budget import Action, BudgetPolicy, CostMeter, Scope
from caliper.pricing import ModelRate, PriceBook


def test_meter_accumulates_across_scopes():
    meter = CostMeter()
    meter.record(100, 50, 0.01)
    meter.record(200, 100, 0.02)
    for scope in Scope:
        u = meter.usage(scope)
        assert u.input_tokens == 300
        assert u.output_tokens == 150
        assert u.calls == 2
        assert round(u.usd, 6) == 0.03


def test_reset_run_preserves_fleet():
    meter = CostMeter()
    meter.record(100, 100, 1.0)
    meter.reset(Scope.RUN)
    assert meter.usage(Scope.RUN).usd == 0.0
    assert meter.usage(Scope.FLEET).usd == 1.0  # fleet total survives a run reset


def test_pricebook_prefix_match():
    pb = PriceBook(rates={"gpt-4o": ModelRate(2.5, 10.0)})
    # exact and versioned both resolve to the gpt-4o rate
    assert pb.cost("gpt-4o", 1_000_000, 0) == 2.5
    assert pb.cost("gpt-4o-2024-08-06", 1_000_000, 0) == 2.5
    # unknown model falls back to zero, not a silent inflation
    assert pb.cost("mystery-model", 1_000_000, 1_000_000) == 0.0


def test_hard_ceiling_halts():
    policy = BudgetPolicy(run_usd_hard=1.0)
    meter = CostMeter()
    meter.record(0, 0, 0.99)
    assert policy.evaluate(meter) is None
    meter.record(0, 0, 0.02)  # now 1.01 >= 1.0
    breach = policy.evaluate(meter)
    assert breach is not None
    assert breach.action is Action.HALT
    assert breach.scope is Scope.RUN


def test_soft_ceiling_downgrades_before_hard():
    policy = BudgetPolicy(run_usd_soft=0.50, run_usd_hard=1.0)
    meter = CostMeter()
    meter.record(0, 0, 0.60)
    breach = policy.evaluate(meter)
    assert breach is not None and breach.action is Action.DOWNGRADE


def test_fleet_ceiling_outranks_run():
    # Each run is small, but the fleet total has blown its ceiling.
    policy = BudgetPolicy(run_usd_hard=10.0, fleet_usd_hard=5.0)
    meter = CostMeter()
    meter.record(0, 0, 6.0)
    breach = policy.evaluate(meter)
    assert breach is not None
    assert breach.scope is Scope.FLEET
    assert breach.action is Action.HALT


def test_token_ceiling():
    policy = BudgetPolicy(run_tokens_hard=1000)
    meter = CostMeter()
    meter.record(600, 500, 0.0)  # 1100 tokens
    breach = policy.evaluate(meter)
    assert breach is not None and breach.limit_tokens == 1000
