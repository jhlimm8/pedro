"""LangGraph wiring for the three sub-graphs: Plan, Plan+, Research.

Each graph shares the same `GraphState` shape so a session can hand off state
between them (most importantly: the approved_plan from a planning graph is
fed into the research graph).

All nodes read deps from `config["configurable"]["deps"]`. The graphs accept
a checkpointer at compile time so chat-turn interrupts can survive across
HTTP calls (FastAPI passes the same thread_id and a shared InMemorySaver).
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from pedro.agents.nodes.dispatcher import dispatcher_node
from pedro.agents.nodes.gates import approval_gate_node, clarify_gate_node
from pedro.agents.nodes.planner_cold import planner_cold_node
from pedro.agents.nodes.planner_plus import planner_plus_node
from pedro.agents.nodes.scout import scout_node
from pedro.agents.nodes.synthesizer import synthesizer_node
from pedro.agents.state import GraphState


def _approval_router(state: GraphState) -> Literal["approved", "loop"]:
    return "approved" if state.get("approved_plan") is not None else "loop"


def _plus_after_planner(state: GraphState) -> Literal["clarify", "approval"]:
    return "clarify" if state.get("awaiting_clarification") else "approval"


# ---------------------------------------------------------------------------
# Plan mode (cold)
# ---------------------------------------------------------------------------


def build_plan_graph(checkpointer=None):
    """Plan mode: cold planner -> approval gate. Baseline / control."""
    g: StateGraph = StateGraph(GraphState)
    g.add_node("planner", planner_cold_node)
    g.add_node("approval", approval_gate_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "approval")
    g.add_conditional_edges(
        "approval",
        _approval_router,
        {"approved": END, "loop": "planner"},
    )
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Plan+ mode (scout-informed)
# ---------------------------------------------------------------------------


def build_plus_graph(checkpointer=None):
    """Plan+ mode: scout -> planner_plus -> [clarify | approval] -> loop or END."""
    g: StateGraph = StateGraph(GraphState)
    g.add_node("scout", scout_node)
    g.add_node("planner", planner_plus_node)
    g.add_node("clarify", clarify_gate_node)
    g.add_node("approval", approval_gate_node)

    g.add_edge(START, "scout")
    g.add_edge("scout", "planner")
    g.add_conditional_edges(
        "planner",
        _plus_after_planner,
        {"clarify": "clarify", "approval": "approval"},
    )
    g.add_edge("clarify", "planner")
    g.add_conditional_edges(
        "approval",
        _approval_router,
        {"approved": END, "loop": "planner"},
    )
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Research mode
# ---------------------------------------------------------------------------


def build_research_graph(checkpointer=None):
    """Research mode: dispatcher -> synthesizer. No interrupts; runs to completion."""
    g: StateGraph = StateGraph(GraphState)
    g.add_node("dispatcher", dispatcher_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "dispatcher")
    g.add_edge("dispatcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile(checkpointer=checkpointer)
