"""Tests for the three loop-detection modes."""

from caliper.loop_detection import LoopDetector


def test_no_false_positive_on_progress():
    det = LoopDetector()
    # Distinct, forward-moving actions never trip.
    for i in range(20):
        v = det.record("search", {"q": f"query-{i}"})
        assert not v.tripped


def test_immediate_repetition():
    det = LoopDetector(max_repeats=4)
    verdicts = [det.record("fetch", {"url": "https://x"}) for _ in range(4)]
    assert not verdicts[2].tripped
    assert verdicts[3].tripped
    assert verdicts[3].kind == "repetition"


def test_periodic_cycle():
    det = LoopDetector(max_cycles=3, max_cycle_period=6)
    tripped = False
    # A,B,A,B,A,B ... — period-2 cycle.
    for i in range(12):
        v = det.record("step", {"k": i % 2})
        tripped = tripped or v.tripped
        if v.tripped:
            assert v.kind in ("cycle", "oscillation")
            break
    assert tripped


def test_state_oscillation():
    det = LoopDetector(oscillation_min_len=16, oscillation_max_distinct=3)
    # Three states cycled enough to fill the window without growing the set.
    tripped = False
    for i in range(20):
        v = det.record("act", {"s": i % 3})
        tripped = tripped or v.tripped
    assert tripped


def test_reset_clears_window():
    det = LoopDetector(max_repeats=3)
    det.record("x")
    det.record("x")
    det.reset()
    v = det.record("x")
    assert not v.tripped  # window was cleared, so the streak restarts
