"""Exceptions raised when Caliper trips its breaker."""

from __future__ import annotations


class CaliperTripped(Exception):
    """Base class for any condition that halts an agent run.

    Carries a machine-readable ``reason`` and a free-form ``detail`` mapping so
    callers (and LangSmith feedback) can classify *why* a run was stopped.
    """

    reason: str = "caliper_tripped"

    def __init__(self, message: str, **detail: object) -> None:
        super().__init__(message)
        self.detail = detail


class BudgetExceeded(CaliperTripped):
    """Raised when a hard budget ceiling is crossed."""

    reason = "budget_exceeded"


class LoopDetected(CaliperTripped):
    """Raised when the loop detector identifies a pathological cycle."""

    reason = "loop_detected"
