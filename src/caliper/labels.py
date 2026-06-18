"""Label / scope-key helpers for dimensional attribution.

A cost event carries labels — at minimum ``agent`` and ``task``, plus any extra
labels the caller attaches. We attribute spend along these dimensions by keying
usage on *scope keys*: a sorted tuple of ``(dimension, value)`` pairs.

    canonical({"agent": "researcher", "task": "t42"})
        -> (("agent", "researcher"), ("task", "t42"))

Three key shapes matter:
  * single-dimension keys  (("agent", "researcher"),)        -> per-agent rollup
  * single-dimension keys  (("task", "t42"),)                -> per-task rollup
  * the full combination   (("agent", ...), ("task", ...))   -> the exact pair

This keeps attribution composable without enumerating the full power set of
labels (which would explode); per-dimension rollups plus the exact combination
cover per-agent, per-task, and per-(agent,task) budgets and baselines.
"""

from __future__ import annotations

from typing import Mapping, Tuple

ScopeKey = Tuple[Tuple[str, str], ...]


def canonical(labels: Mapping[str, object]) -> ScopeKey:
    """The full-combination scope key for a label set (sorted, stringified)."""
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def single_keys(labels: Mapping[str, object]) -> list[ScopeKey]:
    """One single-dimension scope key per label."""
    return [((str(k), str(v)),) for k, v in labels.items()]


def attribution_keys(labels: Mapping[str, object]) -> list[ScopeKey]:
    """The keys a cost event updates: each single dimension, plus the full combo
    when there is more than one label."""
    keys = single_keys(labels)
    if len(labels) > 1:
        keys.append(canonical(labels))
    return keys


def render(key: ScopeKey) -> str:
    """Human-readable scope key: ``agent=researcher,task=t42`` (or ``global``)."""
    return ",".join(f"{d}={v}" for d, v in key) or "global"
