"""Budget primitives: scopes, usage accounting, and policy ceilings."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum


class Scope(str, Enum):
    """Budget scopes, nested coarsest-last.

    A single LLM call is a STEP. One agent invocation (many steps) is a RUN.
    A user/engagement thread of many runs is a SESSION. The whole deployment is
    the FLEET. A ceiling at any scope bounds everything inside it.
    """

    STEP = "step"
    RUN = "run"
    SESSION = "session"
    FLEET = "fleet"


class Action(str, Enum):
    """What to do when a ceiling is crossed."""

    WARN = "warn"
    DOWNGRADE = "downgrade"
    HALT = "halt"


@dataclass
class Usage:
    """Accumulated usage for one scope."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int, usd: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += usd
        self.calls += 1


class CostMeter:
    """Thread-safe, hierarchical accumulation of usage across scopes.

    The FLEET scope persists for the life of the meter. RUN/SESSION scopes can be
    reset at their boundaries (e.g. ``meter.reset(Scope.RUN)`` when a new agent
    invocation begins) without disturbing the fleet total.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: dict[Scope, Usage] = {s: Usage() for s in Scope}

    def record(self, input_tokens: int, output_tokens: int, usd: float) -> None:
        with self._lock:
            for scope in Scope:
                self._usage[scope].add(input_tokens, output_tokens, usd)

    def usage(self, scope: Scope) -> Usage:
        with self._lock:
            u = self._usage[scope]
            return Usage(u.input_tokens, u.output_tokens, u.usd, u.calls)

    def reset(self, scope: Scope) -> None:
        """Reset one scope's counters (use at run/session boundaries)."""
        with self._lock:
            self._usage[scope] = Usage()

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                scope.value: {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "usd": round(u.usd, 6),
                    "calls": u.calls,
                }
                for scope, u in self._usage.items()
            }


@dataclass
class Breach:
    """A crossed ceiling: which scope, which limit, and the action to take."""

    scope: Scope
    action: Action
    limit_usd: float | None
    limit_tokens: int | None
    observed_usd: float
    observed_tokens: int

    def message(self) -> str:
        if self.limit_usd is not None:
            return (
                f"{self.scope.value} budget {self.action.value}: "
                f"${self.observed_usd:.4f} >= ${self.limit_usd:.4f}"
            )
        return (
            f"{self.scope.value} budget {self.action.value}: "
            f"{self.observed_tokens} tok >= {self.limit_tokens} tok"
        )


@dataclass
class BudgetPolicy:
    """Declarative soft/hard ceilings per scope.

    Soft ceilings trigger :attr:`Action.DOWNGRADE` (or WARN); hard ceilings
    trigger :attr:`Action.HALT`. Any unset limit is ignored. Token ceilings and
    dollar ceilings can be combined; the first crossed wins.
    """

    # Dollar ceilings
    run_usd_soft: float | None = None
    run_usd_hard: float | None = None
    session_usd_hard: float | None = None
    fleet_usd_hard: float | None = None
    # Token ceilings
    run_tokens_hard: int | None = None
    session_tokens_hard: int | None = None
    # Behavior at the soft ceiling
    soft_action: Action = Action.DOWNGRADE

    def evaluate(self, meter: CostMeter) -> Breach | None:
        """Return the highest-severity breach, or ``None`` if within budget.

        Hard ceilings outrank soft ones; broader scopes outrank narrower ones at
        equal severity, because a fleet halt is the most consequential stop.
        """
        hard_checks: list[tuple[Scope, float | None, int | None]] = [
            (Scope.FLEET, self.fleet_usd_hard, None),
            (Scope.SESSION, self.session_usd_hard, self.session_tokens_hard),
            (Scope.RUN, self.run_usd_hard, self.run_tokens_hard),
        ]
        for scope, usd_limit, tok_limit in hard_checks:
            u = meter.usage(scope)
            if usd_limit is not None and u.usd >= usd_limit:
                return Breach(scope, Action.HALT, usd_limit, None, u.usd, u.total_tokens)
            if tok_limit is not None and u.total_tokens >= tok_limit:
                return Breach(scope, Action.HALT, None, tok_limit, u.usd, u.total_tokens)

        if self.run_usd_soft is not None:
            u = meter.usage(Scope.RUN)
            if u.usd >= self.run_usd_soft:
                return Breach(
                    Scope.RUN, self.soft_action, self.run_usd_soft, None, u.usd, u.total_tokens
                )
        return None
