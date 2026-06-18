"""Working demo: Caliper bounding a real LangGraph agent.

This runs an ACTUAL LangGraph graph through Caliper's real callback handler and
conditional-edge gating — no mocks of Caliper itself. To stay deterministic and
key-free, the *model* is scripted: a tiny BaseChatModel that emits real
``AIMessage`` objects carrying ``usage_metadata`` (so the callback meters true
token costs) and ``tool_calls`` (so the loop detector sees real action
signatures). Everything Caliper does is the genuine code path.

Run:
    python examples/demo.py          # both scenarios
    python examples/demo.py healthy  # just the healthy run
    python examples/demo.py runaway  # just the runaway run

Scenarios:
    healthy   -> a few distinct tool calls, then a final answer. Finishes under
                 budget; Caliper never trips.
    runaway   -> the model loops on the same tool forever. The LOOP DETECTOR
                 halts it (same action signature repeating).
    expensive -> distinct, non-looping steps whose cost climbs until the
                 PER-TASK BUDGET ceiling halts it. Shows the budget guardrail
                 firing independently of loop detection.
"""

from __future__ import annotations

import sys
from typing import Annotated, Any, Iterator, TypedDict

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Make `caliper` importable without installing the package.
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from caliper import (  # noqa: E402
    AttributionBudget,
    BaselineTracker,
    BudgetPolicy,
    BudgetRule,
    Caliper,
    CaliperTripped,
)


class ScriptedChatModel(BaseChatModel):
    """A deterministic chat model that replays a fixed script of turns.

    Each turn is either a tool call ``("tool", name, args)`` or a final answer
    ``("final", text)``. Every turn reports token usage so the callback meters a
    real cost. When the script is exhausted the model returns a final answer
    (used by the runaway scenario, which is designed to be stopped first).
    """

    script: list[tuple]
    input_tokens: int = 1200      # realistic-ish prompt size that grows the bill
    output_tokens: int = 400
    model_name_: str = "gpt-4o"

    # pydantic v2 model config used by BaseChatModel allows extra fields above.
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._turn = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self) -> tuple:
        if self._turn < len(self.script):
            turn = self.script[self._turn]
        else:
            # past the script: keep emitting the last instruction (runaway loop)
            turn = self.script[-1]
        self._turn += 1
        return turn

    def _usage(self, scale: float) -> dict:
        inp = int(self.input_tokens * scale)
        out = int(self.output_tokens * scale)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        turn = self._next()
        # An optional trailing float on a turn scales that call's token usage,
        # letting the 'expensive' scenario grow cost without looping.
        scale = turn[-1] if isinstance(turn[-1], (int, float)) else 1.0
        if turn[0] == "tool":
            name, args = turn[1], turn[2]
            msg = AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call_{self._turn}"}],
                usage_metadata=self._usage(scale),
            )
        else:
            text = turn[1]
            msg = AIMessage(content=text, usage_metadata=self._usage(scale))
        # llm_output carries the model name so the PriceBook can price the call.
        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output={"model_name": self.model_name_},
        )


# --- the tools (trivial, deterministic) -------------------------------------
def search(query: str) -> str:
    return f"results for {query!r}"


def fetch(url: str) -> str:
    return f"contents of {url}"


TOOLS = {"search": search, "fetch": fetch}


# --- the graph --------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, lambda a, b: a + b]


def build(scenario: str):
    from langgraph.graph import END, START, StateGraph

    if scenario == "healthy":
        script = [
            ("tool", "search", {"query": "langgraph cost control"}),
            ("tool", "fetch", {"url": "https://example.com/a"}),
            ("tool", "search", {"query": "budget enforcement patterns"}),
            ("final", "Here is the synthesized answer."),
        ]
        # Generous ceilings: a healthy run should finish without tripping.
        policy = BudgetPolicy(run_usd_hard=1.00)
        attribution = AttributionBudget([BudgetRule(per="task", usd_hard=1.00)])
    elif scenario == "runaway":  # same tool call forever -> loop detector
        script = [("tool", "fetch", {"url": "https://example.com/stuck"})]
        policy = BudgetPolicy(run_usd_hard=0.20)
        attribution = AttributionBudget([BudgetRule(per="task", usd_hard=0.15)])
    else:  # expensive: distinct steps, climbing cost -> per-task BUDGET halts it
        script = [
            ("tool", "search", {"query": "q1"}, 1.0),
            ("tool", "fetch", {"url": "https://example.com/1"}, 2.0),
            ("tool", "search", {"query": "q2"}, 4.0),
            ("tool", "fetch", {"url": "https://example.com/2"}, 8.0),
            ("tool", "search", {"query": "q3"}, 16.0),  # budget trips around here
            ("final", "should never reach here", 1.0),
        ]
        # Loop detector won't fire (every action signature is distinct); the
        # per-task ceiling is what stops the bleed.
        policy = BudgetPolicy(run_usd_hard=10.0)  # high, so the per-task rule wins
        attribution = AttributionBudget([BudgetRule(per="task", usd_hard=0.12)])

    alerts: list = []
    caliper = Caliper(
        policy=policy,
        attribution_budget=attribution,
        baselines=BaselineTracker(min_samples=3, z_threshold=3.0),
        default_model="gpt-4o",
        on_alert=lambda a: alerts.append(a) or print(f"   ALERT {a}"),
    )

    model = ScriptedChatModel(script=script)
    metered = model.with_config(
        callbacks=[caliper.callback_handler()],
        metadata={"agent": "researcher", "task": f"{scenario}-1"},
    )

    def agent_node(state: State):
        caliper.record_step(state)  # raises LoopDetected on a pathological loop
        return {"messages": [metered.invoke(state["messages"])]}

    def tool_node(state: State):
        last = state["messages"][-1]
        out = []
        for call in last.tool_calls:
            result = TOOLS[call["name"]](**call["args"])
            out.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return {"messages": out}

    def route(state: State):
        if caliper.budget_edge(state) == "halt":
            return END
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    # recursion_limit is LangGraph's own blunt backstop; set high so Caliper is
    # demonstrably what stops the runaway, not the framework's safety net.
    return g.compile(), caliper


def run(scenario: str) -> None:
    print(f"\n=== scenario: {scenario} ===")
    graph, caliper = build(scenario)
    try:
        result = graph.invoke(
            {"messages": [("user", "Research and answer.")]},
            {"recursion_limit": 100},
        )
        final = result["messages"][-1]
        print(f"   COMPLETED: {final.content!r}")
    except CaliperTripped as e:
        print(f"   HALTED by Caliper: {e}  (reason={e.reason})")
    except Exception as e:  # e.g. LangGraph recursion limit — should NOT win the race
        print(f"   stopped by framework backstop, not Caliper: {type(e).__name__}: {e}")
    usage = caliper.snapshot()
    run_usd = usage["usage"]["run"]["usd"]
    print(f"   run spend: ${run_usd:.4f} | per-scope: {usage.get('attributed', {})}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    valid = ("healthy", "runaway", "expensive")
    scenarios = [which] if which in valid else list(valid)
    print("Caliper working demo — real LangGraph graph, scripted key-free model")
    for s in scenarios:
        run(s)
    print()


if __name__ == "__main__":
    main()
