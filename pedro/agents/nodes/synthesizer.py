"""Synthesizer node — produces the final cited markdown report."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from pedro.agents.deps import deps_from_config
from pedro.agents.prompts.synthesizer import (
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_TEMPLATE,
    render_findings_grouped,
    render_plan_sub_questions,
)
from pedro.agents.schemas import SynthesizerOutput
from pedro.agents.state import GraphState
from pedro.api.sse import (
    FinalReportEvent,
    SynthesisStartedEvent,
)


async def synthesizer_node(state: GraphState, config: RunnableConfig) -> dict:
    deps = deps_from_config(config)
    plan = state.get("approved_plan")
    deep_findings = state.get("deep_findings", []) or []

    if plan is None:
        return {"error": "synthesizer: no approved_plan"}
    if not deep_findings:
        return {"error": "synthesizer: no deep_findings"}

    await deps.emit(SynthesisStartedEvent())

    response = await deps.llm.parse(
        instructions=SYNTHESIZER_SYSTEM_PROMPT,
        user_input=SYNTHESIZER_USER_TEMPLATE.format(
            interpreted_intent=plan.interpreted_intent,
            expected_deliverable=plan.expected_deliverable or "(unspecified)",
            plan_sub_questions_block=render_plan_sub_questions(plan.sub_questions),
            findings_block=render_findings_grouped(deep_findings),
        ),
        text_format=SynthesizerOutput,
        with_web_search=False,
    )
    out = response.parsed

    await deps.emit(
        FinalReportEvent(
            report_markdown=out.report_markdown,
            contradictions=out.contradictions,
        )
    )

    return {
        "report_markdown": out.report_markdown,
        "contradictions": out.contradictions,
    }
