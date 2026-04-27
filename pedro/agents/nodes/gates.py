"""Human-in-loop gate nodes: clarification turn and plan approval.

Each gate calls `interrupt()` to pause the graph until the FastAPI route
resumes it via `Command(resume=...)`. The resume payload shapes are:

  approval gate:
    {"action": "approve"}                           -> sets approved_plan
    {"action": "edit", "edits": "<free-text>"}      -> appends to chat_history,
                                                       clears proposed_plan,
                                                       loops back to planner

  clarify gate:
    {"text": "<user reply>"}                        -> appends to chat_history,
                                                       loops back to planner_plus
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from pedro.agents.deps import deps_from_config
from pedro.agents.schemas import ChatTurn
from pedro.agents.state import GraphState
from pedro.api.sse import (
    AssistantMessageEvent,
    ModeLockedEvent,
    PlanApprovedEvent,
)


async def approval_gate_node(state: GraphState, config: RunnableConfig) -> dict:
    deps = deps_from_config(config)
    plan = state.get("proposed_plan")
    if plan is None:
        return {"error": "approval_gate: no proposed_plan in state"}

    user_action = interrupt(
        {
            "kind": "approval",
            "plan": plan.model_dump(),
            "plan_markdown": plan.to_markdown(),
        }
    )

    action = (user_action or {}).get("action", "approve")

    if action == "edit":
        edit_text = (user_action or {}).get("edits", "")
        await deps.emit(
            AssistantMessageEvent(
                content="**Got your edits.** Re-drafting the plan with that direction in mind."
            )
        )
        return {
            "chat_history": [ChatTurn(role="user", content=edit_text)],
            "proposed_plan": None,
            "awaiting_clarification": False,
        }

    await deps.emit(PlanApprovedEvent())
    await deps.emit(ModeLockedEvent())
    return {"approved_plan": plan}


async def clarify_gate_node(state: GraphState, config: RunnableConfig) -> dict:
    if not state.get("awaiting_clarification"):
        # Defensive — this node should only be reached if planner asked.
        return {"awaiting_clarification": False}

    pending = state.get("pending_clarifications") or []
    user_response = interrupt(
        {
            "kind": "clarification",
            "questions": [q.model_dump() for q in pending],
        }
    )

    if isinstance(user_response, dict):
        text = user_response.get("text", "")
    elif isinstance(user_response, str):
        text = user_response
    else:
        text = ""

    return {
        "chat_history": [ChatTurn(role="user", content=text)],
        "awaiting_clarification": False,
        "pending_clarifications": [],
    }
