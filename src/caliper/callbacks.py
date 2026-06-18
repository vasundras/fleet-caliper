"""LangChain callback that meters every LLM call and trips the breaker.

This is the framework-agnostic metering path: anywhere LangChain callbacks are
honored (LCEL, agents, LangGraph nodes that call a model via ``.with_config``),
attaching this handler accounts the cost in real time and raises
:class:`BudgetExceeded` the instant a hard ceiling is crossed — inside the call
stack, so the run actually stops rather than logging an overage after the fact.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .budget import Action, BudgetPolicy, CostMeter
from .exceptions import BudgetExceeded
from .pricing import PriceBook


def _extract_tokens(response: LLMResult) -> tuple[int, int]:
    """Pull (input, output) token counts from an LLMResult across shapes.

    Prefers per-message ``usage_metadata`` (chat models), falls back to the
    aggregate ``llm_output['token_usage']`` (older/completion shapes).
    """
    inp = out = 0
    found = False
    for gens in response.generations:
        for gen in gens:
            msg = getattr(gen, "message", None)
            um = getattr(msg, "usage_metadata", None) if msg is not None else None
            if um:
                inp += int(um.get("input_tokens", 0))
                out += int(um.get("output_tokens", 0))
                found = True
    if found:
        return inp, out

    llm_output = response.llm_output or {}
    tu = llm_output.get("token_usage") or llm_output.get("usage") or {}
    inp = int(tu.get("prompt_tokens", tu.get("input_tokens", 0)))
    out = int(tu.get("completion_tokens", tu.get("output_tokens", 0)))
    return inp, out


def _extract_model(response: LLMResult, default: str | None) -> str | None:
    llm_output = response.llm_output or {}
    return llm_output.get("model_name") or llm_output.get("model") or default


class CaliperCallbackHandler(BaseCallbackHandler):
    """Meters LLM calls into a :class:`CostMeter` and enforces a :class:`BudgetPolicy`.

    Args:
        meter: the shared cost meter (usually owned by a ``Caliper``).
        policy: ceilings to enforce.
        pricebook: token->dollar mapping.
        default_model: used when the result doesn't report its model name.
        on_downgrade: optional callback invoked with a :class:`Breach` when a soft
            ceiling trips, so the caller can swap to a cheaper model or trim tools.
    """

    raise_error = True  # let BudgetExceeded propagate out of the call stack

    def __init__(
        self,
        meter: CostMeter,
        policy: BudgetPolicy,
        pricebook: PriceBook,
        default_model: str | None = None,
        on_downgrade: Any = None,
    ) -> None:
        self.meter = meter
        self.policy = policy
        self.pricebook = pricebook
        self.default_model = default_model
        self.on_downgrade = on_downgrade

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        inp, out = _extract_tokens(response)
        model = _extract_model(response, self.default_model)
        usd = self.pricebook.cost(model, inp, out)
        self.meter.record(inp, out, usd)

        breach = self.policy.evaluate(self.meter)
        if breach is None:
            return
        if breach.action is Action.HALT:
            raise BudgetExceeded(
                breach.message(),
                scope=breach.scope.value,
                observed_usd=breach.observed_usd,
                limit_usd=breach.limit_usd,
            )
        if breach.action is Action.DOWNGRADE and self.on_downgrade is not None:
            self.on_downgrade(breach)
