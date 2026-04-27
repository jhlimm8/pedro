"""The shared researcher primitive.

Used in two phases:
  - "scout": broad+shallow, called from the scout node for each scout sub-question.
  - "deep":  focused+thorough, called from the dispatcher for each plan sub-question.

Same primitive, different parameters. This reuse is the structural payoff of
PEDRO's scout-first thesis: a scout *is* a researcher, just with different
prompts and budgets.

This module exposes a pure async function `research_one(...)` rather than a
LangGraph node directly, because the function is invoked inside fan-out nodes
(scout, dispatcher) which call it concurrently across N sub-questions.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pedro.agents.deps import Deps
from pedro.agents.prompts.researcher import (
    DEEP_SYSTEM_PROMPT,
    DEEP_USER_TEMPLATE,
    SCOUT_SYSTEM_PROMPT,
    SCOUT_USER_TEMPLATE,
)
from pedro.agents.schemas import (
    Finding,
    ResearcherOutput,
    SubQuestion,
)
from pedro.api.sse import (
    DeepFindingEvent,
    ScoutFindingEvent,
    SourceFoundEvent,
    SubQuestionProgressEvent,
)

Phase = Literal["scout", "deep"]


def _format_motivating_findings(findings: list[Finding]) -> str:
    if not findings:
        return "(none)"
    parts: list[str] = []
    for f in findings:
        urls = ", ".join(s.url for s in f.sources) or "no-source"
        parts.append(f"- [{f.id}] {f.headline} ({urls})")
    return "\n".join(parts)


async def research_one(
    *,
    deps: Deps,
    phase: Phase,
    overall_query: str,
    interpreted_intent: str,
    sub_question: SubQuestion,
    motivating_findings: list[Finding] | None = None,
) -> list[Finding]:
    """Execute one researcher call for one sub-question. Returns its Findings.

    The findings are guaranteed to have `sub_question_id` set to the given
    sub_question.id (we coerce in case the model forgets).
    """
    if phase == "scout":
        instructions = SCOUT_SYSTEM_PROMPT
        user_input = SCOUT_USER_TEMPLATE.format(
            overall_query=overall_query,
            sub_question_id=sub_question.id,
            question=sub_question.question,
        )
    else:
        instructions = DEEP_SYSTEM_PROMPT
        user_input = DEEP_USER_TEMPLATE.format(
            interpreted_intent=interpreted_intent or overall_query,
            sub_question_id=sub_question.id,
            depth=sub_question.depth,
            question=sub_question.question,
            rationale=sub_question.rationale or "(none)",
            success_criteria=sub_question.success_criteria or "(none)",
            motivating_findings_block=_format_motivating_findings(
                motivating_findings or []
            ),
        )

    await deps.emit(
        SubQuestionProgressEvent(sub_question_id=sub_question.id, status="started")
    )

    response = await deps.llm.parse(
        instructions=instructions,
        user_input=user_input,
        text_format=ResearcherOutput,
        with_web_search=True,
    )

    findings: list[Finding] = []
    for f in response.parsed.findings:
        # Coerce sub_question_id in case the model omits/mangles it.
        f.sub_question_id = sub_question.id
        findings.append(f)

        if phase == "scout":
            await deps.emit(ScoutFindingEvent(finding=f))
        else:
            await deps.emit(DeepFindingEvent(finding=f))
            for src in f.sources:
                await deps.emit(
                    SourceFoundEvent(sub_question_id=sub_question.id, source=src)
                )

    await deps.emit(
        SubQuestionProgressEvent(sub_question_id=sub_question.id, status="completed")
    )
    return findings


async def research_many(
    *,
    deps: Deps,
    phase: Phase,
    overall_query: str,
    interpreted_intent: str,
    sub_questions: list[SubQuestion],
    motivating_findings_by_sq: dict[str, list[Finding]] | None = None,
    concurrency: int = 4,
) -> list[Finding]:
    """Fan-out helper: concurrently research many sub-questions.

    Used by both the scout node and the deep dispatcher node.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(sq: SubQuestion) -> list[Finding]:
        async with semaphore:
            motivating = (
                (motivating_findings_by_sq or {}).get(sq.id, []) if phase == "deep" else []
            )
            return await research_one(
                deps=deps,
                phase=phase,
                overall_query=overall_query,
                interpreted_intent=interpreted_intent,
                sub_question=sq,
                motivating_findings=motivating,
            )

    results = await asyncio.gather(*(one(sq) for sq in sub_questions))
    flat: list[Finding] = []
    for batch in results:
        flat.extend(batch)
    return flat
