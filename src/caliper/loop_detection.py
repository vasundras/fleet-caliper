"""Online detection of the pathologies that quietly burn agent budget.

Three distinct failure modes, each cheap to detect over a sliding window of
*action signatures* (a stable hash of "what the agent just did" — typically a
tool name plus its normalized arguments):

1. **Immediate repetition** — the same action N times in a row (a stuck retry).
2. **Periodic cycle** — a repeating block of period p that recurs r times at the
   tail of the window (A,B,A,B,A,B ...). Catches multi-step oscillation that
   immediate-repetition checks miss.
3. **State oscillation** — the agent alternates between a small set of states
   without the set growing, i.e. low distinct-signature count over a full
   window (lots of activity, no progress).

None of these require knowing the agent's goal; they are structural. They are
signals, not proofs — the thresholds are tunable and meant to trip *before* a
dollar ceiling does, so a loop is named as a loop rather than mislabeled as
generic overspend.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass


def signature(action: str, args: object = None) -> str:
    """Stable signature for an action + its arguments."""
    try:
        payload = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(args)
    return hashlib.sha1(f"{action}::{payload}".encode()).hexdigest()[:16]


@dataclass
class LoopVerdict:
    tripped: bool
    kind: str | None = None          # "repetition" | "cycle" | "oscillation"
    detail: str | None = None


class LoopDetector:
    """Sliding-window loop detector.

    Args:
        window: how many recent signatures to retain.
        max_repeats: trip if the same signature occurs this many times in a row.
        max_cycles: trip if a periodic block recurs this many times at the tail.
        max_cycle_period: largest cycle period to search for.
        oscillation_min_len: only test oscillation once the window is at least this full.
        oscillation_max_distinct: trip if distinct signatures over a full window
            is at most this (lots of steps, very few distinct states).
    """

    def __init__(
        self,
        window: int = 24,
        max_repeats: int = 4,
        max_cycles: int = 3,
        max_cycle_period: int = 6,
        oscillation_min_len: int = 16,
        oscillation_max_distinct: int = 3,
    ) -> None:
        self.window = window
        self.max_repeats = max_repeats
        self.max_cycles = max_cycles
        self.max_cycle_period = max_cycle_period
        self.oscillation_min_len = oscillation_min_len
        self.oscillation_max_distinct = oscillation_max_distinct
        self._buf: deque[str] = deque(maxlen=window)

    def record(self, action: str, args: object = None) -> LoopVerdict:
        """Record one step and return the current verdict."""
        self._buf.append(signature(action, args))
        return self.check()

    def record_signature(self, sig: str) -> LoopVerdict:
        self._buf.append(sig)
        return self.check()

    def check(self) -> LoopVerdict:
        buf = list(self._buf)
        if not buf:
            return LoopVerdict(False)

        # 1. Immediate repetition at the tail.
        last = buf[-1]
        run = 0
        for s in reversed(buf):
            if s == last:
                run += 1
            else:
                break
        if run >= self.max_repeats:
            return LoopVerdict(True, "repetition", f"{last} x{run}")

        # 2. Periodic cycle at the tail.
        for period in range(2, min(self.max_cycle_period, len(buf) // 2) + 1):
            block = buf[-period:]
            repeats = 1
            idx = len(buf) - period
            while idx - period >= 0 and buf[idx - period:idx] == block:
                repeats += 1
                idx -= period
            if repeats >= self.max_cycles:
                return LoopVerdict(True, "cycle", f"period={period} x{repeats}")

        # 3. State oscillation over a (near-)full window.
        if len(buf) >= self.oscillation_min_len:
            distinct = len(Counter(buf))
            if distinct <= self.oscillation_max_distinct:
                return LoopVerdict(
                    True, "oscillation", f"{distinct} distinct over {len(buf)} steps"
                )

        return LoopVerdict(False)

    def reset(self) -> None:
        self._buf.clear()
