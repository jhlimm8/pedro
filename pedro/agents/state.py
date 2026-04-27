"""LangGraph state for the three sub-graphs.

A single, mode-tagged state type is used across all three graphs so that a
session can transition cleanly from Plan/Plan+ into Research without rebuilding
state. Reducer functions are attached for fields that nodes accumulate to
(chat history, findings, sse events).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from pedro.agents.schemas import (
    ChatTurn,
    ClarifyingQuestion,
    Finding,
    Mode,
    ResearchPlan,
    Source,
    SubQuestion,
)


def append_list(left: list, right: list) -> list:
    """Reducer: concatenate, dedup-by-id when items have an `id` field."""
    if not right:
        return left
    combined = list(left) + list(right)
    seen: set[str] = set()
    out: list = []
    for item in combined:
        item_id = getattr(item, "id", None)
        if item_id is not None:
            if item_id in seen:
                continue
            seen.add(item_id)
        out.append(item)
    return out


class GraphState(TypedDict, total=False):
    """Shared state for Plan / Plan+ / Research sub-graphs.

    `total=False` means every key is optional; nodes only set what they touch.
    """

    # Inputs
    mode: Mode
    user_query: str
    chat_history: Annotated[list[ChatTurn], append_list]

    # Plan+ scout phase
    scout_sub_questions: list[SubQuestion]
    scout_findings: Annotated[list[Finding], append_list]

    # Planning phase (cold or scout-informed)
    proposed_plan: ResearchPlan | None
    planner_thought: str
    awaiting_clarification: bool
    pending_clarifications: list[ClarifyingQuestion]

    # Approval phase
    approved_plan: ResearchPlan | None

    # Research phase
    deep_findings: Annotated[list[Finding], append_list]
    deep_sources: Annotated[list[Source], append_list]

    # Final
    report_markdown: str
    contradictions: list[str]

    # Misc
    error: str | None
