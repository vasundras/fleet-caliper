"""Alerts emitted as cost events are attributed and grounded.

Three kinds, deliberately distinct so consumers can route them differently:

  * **EXHAUSTION** — a budget ceiling for a scope is approaching or crossed.
    ``ratio`` is spend / limit; ``severity`` is "warn" near the soft line and
    "halt" at/over the hard line. This is the paperclip-style signal, now per
    agent and per task rather than only global.
  * **SPIKE** — a single event was statistically anomalous for its scope
    (z-score over threshold). Independent of whether any budget is near.
  * **TREND** — cost-per-event is drifting upward for a scope (short/long EWMA
    ratio over threshold). An early warning before exhaustion.

Alerts are data, not control flow. A hard EXHAUSTION still halts the run via the
existing breaker; SPIKE/TREND are observability by default. Callers attach an
``on_alert`` sink (log, LangSmith feedback, pager, dashboard) to consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .labels import ScopeKey, render


class AlertKind(str, Enum):
    EXHAUSTION = "exhaustion"
    SPIKE = "spike"
    TREND = "trend"


@dataclass
class Alert:
    kind: AlertKind
    scope: ScopeKey
    severity: str               # "info" | "warn" | "halt"
    message: str
    observed_usd: float = 0.0
    detail: dict = field(default_factory=dict)

    @property
    def scope_label(self) -> str:
        return render(self.scope)

    def __str__(self) -> str:
        return f"[{self.kind.value}:{self.severity}] {self.scope_label} :: {self.message}"
