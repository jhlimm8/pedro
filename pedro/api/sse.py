"""Typed SSE event protocol.

Each event is a Pydantic model with a `type` literal discriminator. The JS
client switches on `event.type` to render. Server-side, events are produced
either directly by graph nodes (via a queue passed through the deps) or by
the routes layer (e.g. mode_locked, awaiting_approval).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from pedro.agents.schemas import (
    ClarifyingQuestion,
    Finding,
    Mode,
    ResearchPlan,
    Source,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _SSEBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    seq: int = 0
    ts: datetime = Field(default_factory=_utcnow)


# ---- mode + lifecycle -----------------------------------------------------


class ModeSetEvent(_SSEBase):
    type: Literal["mode_set"] = "mode_set"
    mode: Mode


class ErrorEvent(_SSEBase):
    type: Literal["error"] = "error"
    message: str


class DoneEvent(_SSEBase):
    type: Literal["done"] = "done"


# ---- chat surface ---------------------------------------------------------


class AssistantMessageEvent(_SSEBase):
    """A free-text message from the assistant (markdown). Used for thought
    traces, scout summaries, status narration."""

    type: Literal["assistant_message"] = "assistant_message"
    content: str


# ---- scout phase ----------------------------------------------------------


class ScoutStartedEvent(_SSEBase):
    type: Literal["scout_started"] = "scout_started"
    sub_question_count: int


class ScoutSubQuestionEvent(_SSEBase):
    type: Literal["scout_sub_question"] = "scout_sub_question"
    sub_question_id: str
    question: str


class ScoutFindingEvent(_SSEBase):
    type: Literal["scout_finding"] = "scout_finding"
    finding: Finding


class ScoutCompleteEvent(_SSEBase):
    type: Literal["scout_complete"] = "scout_complete"
    finding_count: int


# ---- planning phase -------------------------------------------------------


class ClarifyingQuestionsEvent(_SSEBase):
    type: Literal["clarifying_questions"] = "clarifying_questions"
    questions: list[ClarifyingQuestion]


class PlanProposedEvent(_SSEBase):
    type: Literal["plan_proposed"] = "plan_proposed"
    plan: ResearchPlan
    plan_markdown: str


class AwaitingApprovalEvent(_SSEBase):
    type: Literal["awaiting_approval"] = "awaiting_approval"


class PlanApprovedEvent(_SSEBase):
    type: Literal["plan_approved"] = "plan_approved"


class ModeLockedEvent(_SSEBase):
    """Emitted when the session locks Plan/Plan+ tabs and auto-switches to Research."""

    type: Literal["mode_locked"] = "mode_locked"


# ---- research phase -------------------------------------------------------


class ResearchStartedEvent(_SSEBase):
    type: Literal["research_started"] = "research_started"
    sub_question_count: int


class SubQuestionProgressEvent(_SSEBase):
    type: Literal["subquestion_progress"] = "subquestion_progress"
    sub_question_id: str
    status: Literal["started", "completed"]


class SourceFoundEvent(_SSEBase):
    type: Literal["source_found"] = "source_found"
    sub_question_id: str
    source: Source


class DeepFindingEvent(_SSEBase):
    type: Literal["deep_finding"] = "deep_finding"
    finding: Finding


class SynthesisStartedEvent(_SSEBase):
    type: Literal["synthesis_started"] = "synthesis_started"


class FinalReportEvent(_SSEBase):
    type: Literal["final_report"] = "final_report"
    report_markdown: str
    contradictions: list[str] = Field(default_factory=list)


# ---- discriminated union --------------------------------------------------


SSEEvent = Annotated[
    Union[
        ModeSetEvent,
        ErrorEvent,
        DoneEvent,
        AssistantMessageEvent,
        ScoutStartedEvent,
        ScoutSubQuestionEvent,
        ScoutFindingEvent,
        ScoutCompleteEvent,
        ClarifyingQuestionsEvent,
        PlanProposedEvent,
        AwaitingApprovalEvent,
        PlanApprovedEvent,
        ModeLockedEvent,
        ResearchStartedEvent,
        SubQuestionProgressEvent,
        SourceFoundEvent,
        DeepFindingEvent,
        SynthesisStartedEvent,
        FinalReportEvent,
    ],
    Field(discriminator="type"),
]


def serialize_sse(event: BaseModel) -> dict[str, str]:
    """Serialize an SSE event for sse-starlette's EventSourceResponse.

    Returns a dict with keys 'event' (the type) and 'data' (json payload).
    """
    payload = event.model_dump(mode="json")
    return {
        "event": payload["type"],
        "data": json.dumps(payload, default=str),
    }
