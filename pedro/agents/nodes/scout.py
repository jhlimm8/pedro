"""Scout node: broad+shallow field reconnaissance for Plan+ mode.

Two-step:
  1. LLM call (no web search) to decompose the user query into N broad
     scout sub-questions and produce an interpretation.
  2. Concurrent fan-out via the shared `research_many` primitive in
     phase="scout" to gather findings across all sub-questions.

Returns partial state with `scout_sub_questions` and `scout_findings`.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from pedro.agents.deps import deps_from_config
from pedro.agents.nodes.researcher import research_many
from pedro.agents.prompts.scout import (
    SCOUT_PLANNER_SYSTEM_PROMPT,
    SCOUT_PLANNER_USER_TEMPLATE,
)
from pedro.agents.schemas import ScoutPlanOutput, SubQuestion
from pedro.agents.state import GraphState
from pedro.api.sse import (
    AssistantMessageEvent,
    ScoutCompleteEvent,
    ScoutStartedEvent,
    ScoutSubQuestionEvent,
)


async def scout_node(state: GraphState, config: RunnableConfig) -> dict:
    deps = deps_from_config(config)
    user_query = state.get("user_query", "")
    if not user_query:
        return {"error": "scout_node: missing user_query in state"}

    min_q = max(2, deps.settings.scout_max_subquestions - 2)
    max_q = deps.settings.scout_max_subquestions

    # ---- Step 1: plan the scout sub-questions
    plan_response = await deps.llm.parse(
        instructions=SCOUT_PLANNER_SYSTEM_PROMPT.format(min_q=min_q, max_q=max_q),
        user_input=SCOUT_PLANNER_USER_TEMPLATE.format(
            user_query=user_query, min_q=min_q, max_q=max_q
        ),
        text_format=ScoutPlanOutput,
        with_web_search=False,
    )
    scout_plan = plan_response.parsed

    if not scout_plan.sub_questions:
        return {"error": "scout_node: planner returned no sub-questions"}

    # Convert ScoutSubQuestion -> SubQuestion (the unified type used downstream).
    sub_qs: list[SubQuestion] = [
        SubQuestion(
            id=ssq.id or f"ssq-{i+1}",
            question=ssq.question,
            rationale=ssq.rationale,
            depth="shallow",
        )
        for i, ssq in enumerate(scout_plan.sub_questions)
    ]

    await deps.emit(
        AssistantMessageEvent(
            content=(
                f"**Scouting the field.** {scout_plan.interpretation}\n\n"
                f"I'll explore {len(sub_qs)} broad angle(s) before drafting a plan."
            )
        )
    )
    await deps.emit(ScoutStartedEvent(sub_question_count=len(sub_qs)))
    for sq in sub_qs:
        await deps.emit(
            ScoutSubQuestionEvent(sub_question_id=sq.id, question=sq.question)
        )

    # ---- Step 2: concurrently scout each sub-question
    findings = await research_many(
        deps=deps,
        phase="scout",
        overall_query=user_query,
        interpreted_intent=scout_plan.interpretation,
        sub_questions=sub_qs,
        concurrency=4,
    )

    await deps.emit(ScoutCompleteEvent(finding_count=len(findings)))

    return {
        "scout_sub_questions": sub_qs,
        "scout_findings": findings,
    }
