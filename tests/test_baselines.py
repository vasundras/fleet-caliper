"""Tests for statistical grounding: per-scope spike and trend detection."""

from caliper.baselines import BaselineTracker


def test_warmup_does_not_judge():
    bt = BaselineTracker(min_samples=8)
    key = (("agent", "a1"),)
    for _ in range(7):
        v = bt.score(key, 0.01)
        assert not v.anomalous  # still warming up
    assert bt.stats_for(key).welford.n == 7


def test_spike_detected_after_baseline():
    bt = BaselineTracker(z_threshold=3.0, min_samples=8)
    key = (("task", "t1"),)
    # Stable baseline around $0.01 with tiny noise.
    for i in range(20):
        bt.score(key, 0.010 + (0.0001 if i % 2 else -0.0001))
    spike = bt.score(key, 0.50)  # 50x the norm
    assert spike.is_spike
    assert spike.zscore > 3.0


def test_no_spike_on_normal_event():
    bt = BaselineTracker(z_threshold=3.0, min_samples=8)
    key = (("task", "t1"),)
    for _ in range(20):
        bt.score(key, 0.010)
    v = bt.score(key, 0.0105)  # within noise
    assert not v.is_spike


def test_trend_detected_on_upward_drift():
    bt = BaselineTracker(trend_ratio=1.5, min_samples=8)
    key = (("agent", "a1"),)
    for _ in range(10):
        bt.score(key, 0.01)
    # Sustained climb pulls the short EWMA well above the long EWMA.
    tripped = False
    for cost in [0.05, 0.08, 0.12, 0.18, 0.25]:
        v = bt.score(key, cost)
        tripped = tripped or v.is_trend
    assert tripped


def test_scopes_are_independent():
    bt = BaselineTracker(z_threshold=3.0, min_samples=4)
    a, b = (("agent", "a1"),), (("agent", "a2"),)
    for _ in range(10):
        bt.score(a, 0.01)
        bt.score(b, 1.00)
    # $0.50 is a spike for the cheap agent, normal for the expensive one.
    assert bt.score(a, 0.50).is_spike
    assert not bt.score(b, 1.00).is_spike
