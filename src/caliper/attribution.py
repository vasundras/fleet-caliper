"""Dimensional attribution: labeled metering and composable per-scope budgets.

This is the axis paperclip lacked. Where :mod:`caliper.budget` meters along the
*temporal* nesting (step/run/session/fleet), this module meters along *labels*
(agent, task, and their combination), so spend is attributable and boundable per
agent, per task, or per exact (agent, task) pair — independently and at once.

  LabeledMeter   accumulates Usage per scope key.
  BudgetRule     a soft/hard ceiling that applies to every scope matching a
                 dimension (e.g. "every agent gets $2 hard") or to one specific
                 value ("the 'researcher' agent gets $5 hard").
  AttributionBudget  a collection of rules, evaluated against a LabeledMeter.

Rules compose: a single event can simultaneously be within its per-agent budget,
over its per-task budget, and trip a per-(agent,task) ceiling. Evaluation returns
all breaches so the caller sees every dimension that fired.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Mapping

from .budget import Action, Usage
from .labels import ScopeKey, attribution_keys, canonical, render


class LabeledMeter:
    """Thread-safe accumulation of usage keyed by scope key.

    A single ``record`` updates every attribution key for the event's labels:
    each single dimension (per-agent, per-task) and the full combination.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: dict[ScopeKey, Usage] = {}

    def record(
        self, labels: Mapping[str, object], input_tokens: int, output_tokens: int, usd: float
    ) -> list[ScopeKey]:
        keys = attribution_keys(labels)
        with self._lock:
            for key in keys:
                u = self._usage.get(key)
                if u is None:
                    u = self._usage[key] = Usage()
                u.add(input_tokens, output_tokens, usd)
        return keys

    def usage(self, key: ScopeKey) -> Usage:
        with self._lock:
            u = self._usage.get(key)
            return Usage(u.input_tokens, u.output_tokens, u.usd, u.calls) if u else Usage()

    def usage_for_labels(self, labels: Mapping[str, object]) -> Usage:
        return self.usage(canonical(labels))

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                render(key): {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "usd": round(u.usd, 6),
                    "calls": u.calls,
                }
                for key, u in self._usage.items()
            }


@dataclass(frozen=True)
class BudgetRule:
    """A ceiling that applies to scopes along one dimension.

    Args:
        per: the dimension this rule governs ("agent", "task", ...).
        usd_soft / usd_hard: dollar ceilings (soft -> downgrade, hard -> halt).
        tokens_hard: optional token ceiling.
        match: if given, the rule applies only to this value of ``per``
            (e.g. per="agent", match="researcher"); if None, it applies to every
            value of that dimension independently.
    """

    per: str
    usd_soft: float | None = None
    usd_hard: float | None = None
    tokens_hard: int | None = None
    match: str | None = None

    def applies_to(self, key: ScopeKey) -> bool:
        """True if ``key`` is a single-dimension key on this rule's dimension
        (and matches ``match`` when set)."""
        if len(key) != 1:
            return False
        dim, val = key[0]
        if dim != self.per:
            return False
        return self.match is None or val == self.match


@dataclass
class AttributionBreach:
    rule: BudgetRule
    scope: ScopeKey
    action: Action
    limit_usd: float | None
    limit_tokens: int | None
    observed_usd: float
    observed_tokens: int

    def message(self) -> str:
        if self.limit_usd is not None:
            return (
                f"{render(self.scope)} {self.action.value}: "
                f"${self.observed_usd:.4f} >= ${self.limit_usd:.4f}"
            )
        return (
            f"{render(self.scope)} {self.action.value}: "
            f"{self.observed_tokens} tok >= {self.limit_tokens} tok"
        )


@dataclass
class AttributionBudget:
    """A set of composable per-dimension rules."""

    rules: list[BudgetRule] = field(default_factory=list)

    def add(self, rule: BudgetRule) -> "AttributionBudget":
        self.rules.append(rule)
        return self

    def evaluate(self, meter: LabeledMeter, touched: list[ScopeKey]) -> list[AttributionBreach]:
        """Return every breach across the scope keys touched by the last event.

        Checks hard ceilings first (HALT), then soft (DOWNGRADE), per rule. All
        firing breaches are returned so the caller sees each dimension that
        tripped, not just the first.
        """
        breaches: list[AttributionBreach] = []
        for key in touched:
            u = meter.usage(key)
            for rule in self.rules:
                if not rule.applies_to(key):
                    continue
                if rule.usd_hard is not None and u.usd >= rule.usd_hard:
                    breaches.append(
                        AttributionBreach(
                            rule, key, Action.HALT, rule.usd_hard, None, u.usd, u.total_tokens
                        )
                    )
                elif rule.tokens_hard is not None and u.total_tokens >= rule.tokens_hard:
                    breaches.append(
                        AttributionBreach(
                            rule, key, Action.HALT, None, rule.tokens_hard, u.usd, u.total_tokens
                        )
                    )
                elif rule.usd_soft is not None and u.usd >= rule.usd_soft:
                    breaches.append(
                        AttributionBreach(
                            rule, key, Action.DOWNGRADE, rule.usd_soft, None, u.usd, u.total_tokens
                        )
                    )
        return breaches
