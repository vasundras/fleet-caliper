"""Statistical baselines for cost events, per attribution scope.

Paperclip-style budgets answer one question: "have we spent too much in total?"
That is necessary but blunt — it only fires at exhaustion, and it cannot tell a
*normal* expensive task from an *anomalous* one. This module adds the second
question: "is this cost event normal for *this agent* and *this task*?"

For every scope key (per-agent, per-task, per-(agent,task)) we maintain online
statistics over per-event cost:

  * **Welford mean/variance** — running mean and standard deviation with no
    stored history, so a z-score is available from the first events on.
  * **Short and long EWMAs** — a fast and a slow exponentially-weighted moving
    average; their ratio is a trend signal (short rising well above long means
    cost-per-event is climbing for this scope).

From these we derive two signals, distinct from budget exhaustion:

  * **SPIKE** — a single event whose z-score exceeds ``z_threshold`` (this call
    cost far more than this scope's norm).
  * **TREND** — short/long EWMA ratio exceeds ``trend_ratio`` (this scope's
    cost-per-event is drifting upward over a run of events).

Both require a warmup (``min_samples``) so we do not cry wolf on the first few
events before a baseline exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _Welford:
    """Online mean/variance (Welford's algorithm)."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class _Ewma:
    """Exponentially-weighted moving average."""

    alpha: float
    value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


@dataclass
class ScopeStats:
    """Per-scope online statistics over per-event cost."""

    welford: _Welford = field(default_factory=_Welford)
    short: _Ewma = field(default_factory=lambda: _Ewma(alpha=0.5))
    long: _Ewma = field(default_factory=lambda: _Ewma(alpha=0.1))

    def update(self, cost: float) -> None:
        self.welford.update(cost)
        self.short.update(cost)
        self.long.update(cost)

    def zscore(self, cost: float) -> float:
        sd = self.welford.stddev
        if sd == 0.0:
            return 0.0
        return (cost - self.welford.mean) / sd

    def is_spike(self, cost: float, z_threshold: float, rel_tol: float) -> tuple[bool, float]:
        """Decide spike, returning (is_spike, zscore).

        Uses the z-score when the baseline has nonzero variance. When variance is
        zero (a perfectly constant history — exactly when a jump is most telling,
        yet z is undefined), fall back to a relative-deviation test: a spike is a
        cost at least ``rel_tol`` above the mean. This keeps small rounding-level
        deltas quiet while still catching a step change off a flat baseline.
        """
        sd = self.welford.stddev
        if sd > 0.0:
            z = (cost - self.welford.mean) / sd
            return z >= z_threshold, z
        mean = self.welford.mean
        if mean > 0.0 and cost >= mean * (1.0 + rel_tol):
            return True, float("inf")
        return False, 0.0

    def trend_ratio(self) -> float:
        if not self.short.value or not self.long.value:
            return 1.0
        return self.short.value / self.long.value


@dataclass
class SpikeVerdict:
    """Outcome of grounding one cost event against a scope baseline."""

    is_spike: bool = False
    is_trend: bool = False
    zscore: float = 0.0
    trend_ratio: float = 1.0
    mean: float = 0.0
    samples: int = 0

    @property
    def anomalous(self) -> bool:
        return self.is_spike or self.is_trend


class BaselineTracker:
    """Maintains a :class:`ScopeStats` per scope key and scores new events.

    Update *after* reading the verdict for an event so the event is scored
    against the baseline as it stood *before* the event — otherwise a large
    event partly masks its own anomaly.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        trend_ratio: float = 1.75,
        min_samples: int = 8,
        rel_tol: float = 0.5,
    ) -> None:
        self.z_threshold = z_threshold
        self.trend_ratio = trend_ratio
        self.min_samples = min_samples
        self.rel_tol = rel_tol  # relative-deviation spike threshold for flat baselines
        self._stats: dict[tuple, ScopeStats] = {}

    def score(self, key: tuple, cost: float) -> SpikeVerdict:
        """Score ``cost`` for scope ``key`` against its baseline, then update it."""
        stats = self._stats.get(key)
        if stats is None:
            stats = self._stats[key] = ScopeStats()

        if stats.welford.n < self.min_samples:
            stats.update(cost)  # warmup: learn, don't judge
            return SpikeVerdict(mean=stats.welford.mean, samples=stats.welford.n)

        is_spike, z = stats.is_spike(cost, self.z_threshold, self.rel_tol)
        ratio = stats.trend_ratio()
        verdict = SpikeVerdict(
            is_spike=is_spike,
            is_trend=ratio >= self.trend_ratio,
            zscore=z,
            trend_ratio=ratio,
            mean=stats.welford.mean,
            samples=stats.welford.n,
        )
        stats.update(cost)
        return verdict

    def stats_for(self, key: tuple) -> ScopeStats | None:
        return self._stats.get(key)
