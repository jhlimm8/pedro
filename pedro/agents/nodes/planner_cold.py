"""Cold planner node — used by Plan mode.

Produces a `ResearchPlan` directly from the user's query without any scouting
or web search. This is the baseline / control for the Plan-vs-Plan+ comparison.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from pedro.agents.deps import deps_from_config
from pedro.agents.prompts.planner import (
    COLD_PLANNER_SYSTEM_PROMPT,
    COLD_PLANNER_USER_TEMPLATE,
)
from pedro.agents.schemas import ResearchPlan
from pedro.agents.state import GraphState
from pedro.api.sse import (
    AssistantMessageEvent,
    PlanProposedEvent,
)


async def planner_cold_node(state: GraphState, config: RunnableConfig) -> dict:
    deps = deps_from_config(config)
    user_query = state.get("user_query", "")
    if not user_query:
        return {"error": "planner_cold: missing user_query"}

    await deps.emit(
        AssistantMessageEvent(
            content="**Drafting plan (Plan mode).** No scouting; cold draft from the query alone."
        )
    )

    response = await deps.llm.parse(
        instructions=COLD_PLANNER_SYSTEM_PROMPT,
        user_input=COLD_PLANNER_USER_TEMPLATE.format(user_query=user_query),
        text_format=ResearchPlan,
        with_web_search=False,
    )
    plan = response.parsed

    await deps.emit(
        PlanProposedEvent(plan=plan, plan_markdown=plan.to_markdown())
    )
    # Note: AwaitingApprovalEvent is emitted by Session._run_planning AFTER the
    # graph actually pauses at the approval gate. Emitting it here would race
    # with the gate's interrupt() — the test/UI could observe the event and
    # call respond() before the graph has paused & checkpointed.

    return {
        "proposed_plan": plan,
        "planner_thought": "(cold draft — no scout findings)",
        "awaiting_clarification": False,
    }
