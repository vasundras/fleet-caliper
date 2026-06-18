"""The Caliper facade: ties metering, policy, pricing, and loop detection together
and exposes the LangGraph integration points."""

from __future__ import annotations

from typing import Any, Literal

from .budget import BudgetPolicy, CostMeter, Scope
from .callbacks import CaliperCallbackHandler
from .exceptions import LoopDetected
from .loop_detection import LoopDetector, signature
from .pricing import PriceBook


class Caliper:
    """One object to bound an agent run economically and behaviorally.

    Typical wiring (LangGraph):

        caliper = Caliper(policy=BudgetPolicy(run_usd_hard=1.0), pricebook=PriceBook.default())
        llm = model.with_config(callbacks=[caliper.callback_handler()])
        graph.add_conditional_edges("agent", caliper.budget_edge, {"continue": "tools", "halt": END})

        def agent_node(state):
            caliper.record_step(state)   # raises LoopDetected on a pathological loop
            ...
    """

    def __init__(
        self,
        policy: BudgetPolicy,
        pricebook: PriceBook | None = None,
        loop_detector: LoopDetector | None = None,
        default_model: str | None = None,
        on_downgrade: Any = None,
    ) -> None:
        self.policy = policy
        self.pricebook = pricebook or PriceBook.default()
        self.meter = CostMeter()
        self.loop_detector = loop_detector or LoopDetector()
        self.default_model = default_model
        self.on_downgrade = on_downgrade

    # --- metering -----------------------------------------------------------
    def callback_handler(self) -> CaliperCallbackHandler:
        """A fresh handler bound to this Caliper's meter and policy."""
        return CaliperCallbackHandler(
            meter=self.meter,
            policy=self.policy,
            pricebook=self.pricebook,
            default_model=self.default_model,
            on_downgrade=self.on_downgrade,
        )

    # --- LangGraph gating ---------------------------------------------------
    def budget_edge(self, state: Any = None) -> Literal["continue", "halt"]:
        """Conditional-edge function: ``"halt"`` once any hard ceiling is crossed."""
        return "halt" if self.policy.evaluate(self.meter) is not None else "continue"

    def record_step(self, state: Any) -> None:
        """Feed the loop detector from graph state; raise on a detected loop.

        Derives an action signature from the most recent tool call in the
        message history when present, otherwise from a hash of the last message.
        Override :meth:`step_signature` for custom state shapes.
        """
        sig = self.step_signature(state)
        if sig is None:
            return
        verdict = self.loop_detector.record_signature(sig)
        if verdict.tripped:
            raise LoopDetected(
                f"loop detected ({verdict.kind}): {verdict.detail}",
                kind=verdict.kind,
                detail=verdict.detail,
            )

    def step_signature(self, state: Any) -> str | None:
        """Best-effort action signature from common LangGraph state shapes."""
        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return None
        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            tc = tool_calls[0]
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            return signature(name, args)
        content = getattr(last, "content", None)
        return signature(type(last).__name__, content)

    # --- observability ------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {"usage": self.meter.snapshot()}

    def reset_run(self) -> None:
        """Reset per-run accounting and the loop window at a run boundary."""
        self.meter.reset(Scope.RUN)
        self.loop_detector.reset()
