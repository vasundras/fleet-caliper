"""LangChain callback that meters every LLM call, attributes it, grounds it
against per-scope baselines, and trips the breaker.

This is the framework-agnostic metering path: anywhere LangChain callbacks are
honored (LCEL, agents, LangGraph nodes that call a model via ``.with_config``),
attaching this handler accounts the cost in real time and raises
:class:`BudgetExceeded` the instant a hard ceiling is crossed — inside the call
stack, so the run actually stops rather than logging an overage after the fact.

Per-event **labels** (``agent``, ``task``, ...) ride on the call's ``metadata``
(set via ``.with_config(metadata={"agent": ..., "task": ...})``). They drive
dimensional attribution and per-agent / per-task baselines.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .alerts import Alert, AlertKind
from .attribution import AttributionBudget, LabeledMeter
from .baselines import BaselineTracker
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
    """Meters LLM calls, attributes them per label, grounds them, and enforces budgets.

    Args:
        meter: temporal cost meter (step/run/session/fleet).
        policy: temporal ceilings to enforce.
        pricebook: token->dollar mapping.
        labeled_meter: optional dimensional meter for per-agent / per-task spend.
        attribution_budget: optional composable per-dimension rules.
        baselines: optional statistical baseline tracker for spike/trend grounding.
        label_keys: which metadata keys are treated as attribution labels.
        default_model: used when the result doesn't report its model name.
        on_downgrade: invoked with a temporal :class:`Breach` at a soft ceiling.
        on_alert: invoked with each :class:`Alert` (exhaustion / spike / trend).
    """

    raise_error = True  # let BudgetExceeded propagate out of the call stack

    def __init__(
        self,
        meter: CostMeter,
        policy: BudgetPolicy,
        pricebook: PriceBook,
        labeled_meter: LabeledMeter | None = None,
        attribution_budget: AttributionBudget | None = None,
        baselines: BaselineTracker | None = None,
        label_keys: tuple[str, ...] = ("agent", "task"),
        default_model: str | None = None,
        on_downgrade: Callable[[Any], None] | None = None,
        on_alert: Callable[[Alert], None] | None = None,
    ) -> None:
        self.meter = meter
        self.policy = policy
        self.pricebook = pricebook
        self.labeled_meter = labeled_meter
        self.attribution_budget = attribution_budget
        self.baselines = baselines
        self.label_keys = label_keys
        self.default_model = default_model
        self.on_downgrade = on_downgrade
        self.on_alert = on_alert
        # metadata is delivered on on_llm_start; stash per run_id for on_llm_end.
        self._labels_by_run: dict[UUID, dict[str, str]] = {}

    # --- label capture ------------------------------------------------------
    def _labels_from(self, metadata: Mapping[str, Any] | None) -> dict[str, str]:
        if not metadata:
            return {}
        return {k: str(metadata[k]) for k in self.label_keys if k in metadata}

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: UUID,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._labels_by_run[run_id] = self._labels_from(metadata)

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: UUID,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._labels_by_run[run_id] = self._labels_from(metadata)

    # --- metering + enforcement --------------------------------------------
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
        labels = self._labels_by_run.pop(run_id, {}) if run_id is not None else {}

        # 1. Temporal metering (step/run/session/fleet).
        self.meter.record(inp, out, usd)

        # 2. Dimensional attribution (per-agent / per-task / per-pair).
        touched: list = []
        if self.labeled_meter is not None and labels:
            touched = self.labeled_meter.record(labels, inp, out, usd)

        # 3. Statistical grounding — spike/trend per scope, before exhaustion.
        if self.baselines is not None and touched and self.on_alert is not None:
            for key in touched:
                v = self.baselines.score(key, usd)
                if v.is_spike:
                    self.on_alert(Alert(
                        AlertKind.SPIKE, key, "warn",
                        f"event ${usd:.4f} is {v.zscore:.1f}σ above mean ${v.mean:.4f}",
                        observed_usd=usd,
                        detail={"zscore": v.zscore, "mean": v.mean, "samples": v.samples},
                    ))
                if v.is_trend:
                    self.on_alert(Alert(
                        AlertKind.TREND, key, "info",
                        f"cost-per-event trending up (short/long={v.trend_ratio:.2f})",
                        observed_usd=usd,
                        detail={"trend_ratio": v.trend_ratio, "samples": v.samples},
                    ))

        # 4. Dimensional budget rules — emit exhaustion alerts; halt on hard.
        if self.attribution_budget is not None and touched:
            for b in self.attribution_budget.evaluate(self.labeled_meter, touched):
                if self.on_alert is not None:
                    sev = "halt" if b.action is Action.HALT else "warn"
                    self.on_alert(Alert(
                        AlertKind.EXHAUSTION, b.scope, sev, b.message(), observed_usd=b.observed_usd,
                    ))
                if b.action is Action.HALT:
                    raise BudgetExceeded(
                        b.message(), scope=b.message(), observed_usd=b.observed_usd,
                        limit_usd=b.limit_usd,
                    )
                if b.action is Action.DOWNGRADE and self.on_downgrade is not None:
                    self.on_downgrade(b)

        # 5. Temporal budget enforcement (existing behavior).
        breach = self.policy.evaluate(self.meter)
        if breach is None:
            return
        if breach.action is Action.HALT:
            if self.on_alert is not None:
                self.on_alert(Alert(
                    AlertKind.EXHAUSTION, (("scope", breach.scope.value),), "halt",
                    breach.message(), observed_usd=breach.observed_usd,
                ))
            raise BudgetExceeded(
                breach.message(), scope=breach.scope.value,
                observed_usd=breach.observed_usd, limit_usd=breach.limit_usd,
            )
        if breach.action is Action.DOWNGRADE and self.on_downgrade is not None:
            self.on_downgrade(breach)
