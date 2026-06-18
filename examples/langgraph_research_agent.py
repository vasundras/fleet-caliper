"""Example: a LangGraph ReAct-style agent bounded by Caliper.

Runnable sketch showing the three integration points:
  1. meter every model call via the callback handler
  2. gate the graph with budget_edge
  3. feed the loop detector with record_step

This file is illustrative — wire in your own model and tools. It avoids hard
dependencies on a specific provider so it reads as documentation even without
an API key.
"""

from __future__ import annotations

from caliper import (
    AttributionBudget,
    BaselineTracker,
    BudgetPolicy,
    BudgetRule,
    Caliper,
    CaliperTripped,
    PriceBook,
)


def build(llm, tools, agent_name: str = "researcher", task_id: str = "task-1"):
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing import Annotated, TypedDict

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    caliper = Caliper(
        policy=BudgetPolicy(
            run_usd_soft=0.50,    # soft: trigger a downgrade hook
            run_usd_hard=1.00,    # hard: halt the run
            fleet_usd_hard=50.0,  # hard: total ceiling across all runs
        ),
        attribution_budget=AttributionBudget([
            BudgetRule(per="agent", usd_hard=2.00),                 # per-agent ceiling
            BudgetRule(per="task", usd_soft=0.40, usd_hard=0.50),   # per-task ceiling
        ]),
        baselines=BaselineTracker(z_threshold=3.0, trend_ratio=1.75),
        pricebook=PriceBook.default(),
        default_model="gpt-4o-mini",
        on_downgrade=lambda breach: print(f"[caliper] soft limit hit: {breach.message()}"),
        on_alert=lambda alert: print(f"[caliper] {alert}"),  # exhaustion / spike / trend
    )

    # Labels on metadata make every model call attributable to an agent and task.
    metered_llm = llm.bind_tools(tools).with_config(
        callbacks=[caliper.callback_handler()],
        metadata={"agent": agent_name, "task": task_id},
    )

    def agent_node(state: State):
        caliper.record_step(state)  # raises LoopDetected on a pathological loop
        return {"messages": [metered_llm.invoke(state["messages"])]}

    def tool_node(state: State):
        last = state["messages"][-1]
        out = []
        for call in last.tool_calls:
            tool = next(t for t in tools if t.name == call["name"])
            out.append(tool.invoke(call))
        return {"messages": out}

    def route(state: State):
        if caliper.budget_edge(state) == "halt":
            return END
        return "tools" if state["messages"][-1].tool_calls else END

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(), caliper


def run(graph, caliper, prompt: str):
    try:
        result = graph.invoke({"messages": [("user", prompt)]})
        print(result["messages"][-1].content)
    except CaliperTripped as e:
        # Halted cleanly on a budget ceiling or a detected loop.
        print(f"[caliper] run halted: {e} (reason={e.reason})")
    finally:
        print("[caliper] usage:", caliper.snapshot()["usage"]["run"])
