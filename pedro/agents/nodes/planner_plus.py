"""Scout-informed planner node — used by Plan+ mode.

Consumes scout findings and chat history, returns either a proposed plan
(with sub-questions attributed to scout finding ids) OR a list of clarifying
questions that should pause the graph for a chat turn.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from pedro.agents.deps import deps_from_config
from pedro.agents.prompts.planner import (
    PLUS_PLANNER_SYSTEM_PROMPT,
    PLUS_PLANNER_USER_TEMPLATE,
    render_chat_history,
    render_scout_findings,
)
from pedro.agents.schemas import PlannerOutput
from pedro.agents.state import GraphState
from pedro.api.sse import (
    AssistantMessageEvent,
    PlanProposedEvent,
)


async def planner_plus_node(state: GraphState, config: RunnableConfig) -> dict:
    deps = deps_from_config(config)
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    scout_findings = state.get("scout_findings", [])

    if not user_query:
        return {"error": "planner_plus: missing user_query"}
    if not scout_findings:
        return {"error": "planner_plus: missing scout_findings (scout must run first)"}

    await deps.emit(
        AssistantMessageEvent(
            content=(
                f"**Drafting plan (Plan+ mode)** from {len(scout_findings)} scout finding(s)."
            )
        )
    )

    response = await deps.llm.parse(
        instructions=PLUS_PLANNER_SYSTEM_PROMPT.format(
            max_q=deps.settings.deep_max_subquestions
        ),
        user_input=PLUS_PLANNER_USER_TEMPLATE.format(
            user_query=user_query,
            chat_history_block=render_chat_history(chat_history),
            scout_findings_block=render_scout_findings(scout_findings),
        ),
        text_format=PlannerOutput,
        with_web_search=False,
    )
    out = response.parsed

    if out.kind == "clarify":
        if not out.questions:
            return {"error": "planner_plus: clarify chosen but no questions provided"}
        await deps.emit(
            AssistantMessageEvent(content=f"**Need to clarify before planning:**\n\n{out.thought}")
        )
        # ClarifyingQuestionsEvent is emitted by Session._run_planning after
        # the graph pauses at clarify_gate (see comment in planner_cold).
        return {
            "proposed_plan": None,
            "planner_thought": out.thought,
            "awaiting_clarification": True,
            "pending_clarifications": list(out.questions),
        }

    if out.plan is None:
        return {"error": "planner_plus: kind=plan but plan is null"}

    await deps.emit(
        AssistantMessageEvent(content=f"**Planner thought process:**\n\n{out.thought}")
    )
    await deps.emit(
        PlanProposedEvent(plan=out.plan, plan_markdown=out.plan.to_markdown())
    )
    # AwaitingApprovalEvent is emitted by Session._run_planning after pause.

    return {
        "proposed_plan": out.plan,
        "planner_thought": out.thought,
        "awaiting_clarification": False,
        "pending_clarifications": [],
    }
