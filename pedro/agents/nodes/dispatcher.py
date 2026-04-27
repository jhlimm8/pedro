"""Deep research dispatcher node — entry to Research mode.

Takes the `approved_plan` and concurrently runs the deep-phase researcher for
each sub-question via the shared `research_many` primitive. Builds a
per-sub-question motivating_findings map by matching sub-question
`motivating_findings` ids back to the session's scout_findings (if any).
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from pedro.agents.deps import deps_from_config
from pedro.agents.nodes.researcher import research_many
from pedro.agents.state import GraphState
from pedro.api.sse import (
    AssistantMessageEvent,
    ResearchStartedEvent,
)


async def dispatcher_node(state: GraphState, config: RunnableConfig) -> dict:
    deps = deps_from_config(config)
    plan = state.get("approved_plan")
    if plan is None:
        return {"error": "dispatcher: no approved_plan in state"}
    if not plan.sub_questions:
        return {"error": "dispatcher: approved_plan has no sub-questions"}

    scout_findings = state.get("scout_findings", []) or []
    findings_by_id = {f.id: f for f in scout_findings}

    motivating_by_sq: dict[str, list] = {}
    for sq in plan.sub_questions:
        motivating_by_sq[sq.id] = [
            findings_by_id[fid]
            for fid in sq.motivating_findings
            if fid in findings_by_id
        ]

    await deps.emit(
        AssistantMessageEvent(
            content=(
                f"**Researching.** Executing {len(plan.sub_questions)} approved sub-question(s) in parallel."
            )
        )
    )
    await deps.emit(ResearchStartedEvent(sub_question_count=len(plan.sub_questions)))

    deep_findings = await research_many(
        deps=deps,
        phase="deep",
        overall_query=state.get("user_query", ""),
        interpreted_intent=plan.interpreted_intent,
        sub_questions=plan.sub_questions,
        motivating_findings_by_sq=motivating_by_sq,
        concurrency=4,
    )

    deep_sources = []
    seen_urls: set[str] = set()
    for f in deep_findings:
        for s in f.sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                deep_sources.append(s)

    return {
        "deep_findings": deep_findings,
        "deep_sources": deep_sources,
    }
