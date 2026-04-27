"""Domain Pydantic schemas shared by the graph nodes, the API, and the tests.

Every schema here is also a candidate for `text_format=` in `responses.parse(...)`,
so we deliberately stick to JSON-friendly types (str over HttpUrl, etc).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Mode = Literal["plan", "plan_plus", "research"]

SourceTier = Literal["official", "reputable", "common"]
"""Source tiers mirror the mdb judge model: official > reputable > common-agreement."""

SubQuestionDepth = Literal["shallow", "medium", "deep"]
TargetSourceType = Literal[
    "official", "academic", "news", "industry", "wiki", "other"
]
ChatRole = Literal["user", "assistant", "system"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Citations & findings
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A web citation. `tier` is filled by the source-tier tagging add-on."""

    model_config = ConfigDict(extra="ignore")

    url: str
    title: str = ""
    tier: SourceTier | None = None


class Finding(BaseModel):
    """A unit of evidence returned by the researcher primitive.

    Used in both phases:
      - scout phase: `sub_question_id` is the scout's broad exploratory question.
      - deep phase:  `sub_question_id` references a plan SubQuestion.id.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    headline: str = Field(description="One-line takeaway.")
    detail: str = Field(description="A paragraph of substantive evidence.")
    sources: list[Source] = Field(default_factory=list)
    sub_question_id: str = Field(default="")


# ---------------------------------------------------------------------------
# Research plan
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    rationale: str = ""
    motivating_findings: list[str] = Field(
        default_factory=list,
        description="IDs of scout Findings that motivate this sub-question. Empty in cold Plan mode.",
    )
    target_source_types: list[TargetSourceType] = Field(default_factory=list)
    depth: SubQuestionDepth = "medium"
    success_criteria: str = ""


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interpreted_intent: str = Field(
        description="The planner's restatement of what the user actually wants."
    )
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    expected_deliverable: str = ""

    def to_markdown(self) -> str:
        """Render the plan as the in-chat artifact users actually read."""
        lines: list[str] = ["# Research Plan", ""]

        lines += ["## Interpreted intent", "", self.interpreted_intent.strip(), ""]

        lines += ["## Scope", "", "**In scope:**"]
        if self.scope_in:
            lines += [f"- {x}" for x in self.scope_in]
        else:
            lines.append("- (none specified)")
        lines += ["", "**Out of scope:**"]
        if self.scope_out:
            lines += [f"- {x}" for x in self.scope_out]
        else:
            lines.append("- (none specified)")
        lines.append("")

        lines += ["## Sub-questions", ""]
        if not self.sub_questions:
            lines += ["_(no sub-questions)_", ""]
        for i, sq in enumerate(self.sub_questions, 1):
            lines += [f"### {i}. {sq.question}", ""]
            if sq.rationale:
                lines += [f"**Rationale:** {sq.rationale}", ""]
            if sq.motivating_findings:
                refs = ", ".join(f"`{fid}`" for fid in sq.motivating_findings)
                lines += [f"**Motivated by:** {refs}", ""]
            meta_bits: list[str] = []
            if sq.target_source_types:
                meta_bits.append("sources: " + ", ".join(sq.target_source_types))
            meta_bits.append(f"depth: {sq.depth}")
            lines += [f"_{' · '.join(meta_bits)}_", ""]
            if sq.success_criteria:
                lines += [f"**Success criteria:** {sq.success_criteria}", ""]

        lines += ["## Expected deliverable", "", self.expected_deliverable.strip()]
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: ChatRole
    content: str
    ts: datetime = Field(default_factory=_utcnow)


class ClarifyingQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    rationale: str = ""


# ---------------------------------------------------------------------------
# LLM-call output schemas (what `responses.parse(text_format=...)` returns)
# ---------------------------------------------------------------------------


class ScoutSubQuestion(BaseModel):
    """A broad/shallow sub-question the scout uses to map the field."""

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    rationale: str = ""


class ScoutPlanOutput(BaseModel):
    """Step 1 of scouting: the scout decomposes the query into N broad sub-questions."""

    model_config = ConfigDict(extra="ignore")

    interpretation: str = Field(
        description="One-paragraph restatement of what the user is asking, before searching."
    )
    sub_questions: list[ScoutSubQuestion] = Field(default_factory=list)


class ResearcherOutput(BaseModel):
    """A single researcher invocation returns N findings for one sub-question."""

    model_config = ConfigDict(extra="ignore")

    findings: list[Finding] = Field(default_factory=list)


class PlannerProposesPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["plan"] = "plan"
    thought: str = Field(
        description="The planner's reasoning trail; visible to the user as 'thought process'."
    )
    plan: ResearchPlan


class PlannerAsksClarification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["clarify"] = "clarify"
    thought: str
    questions: list[ClarifyingQuestion] = Field(min_length=1)


class PlannerOutput(BaseModel):
    """Discriminated union for planner output. We use a single wrapper rather
    than a top-level union because the OpenAI structured-output schema is
    simpler with a single object root."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["plan", "clarify"]
    thought: str
    plan: ResearchPlan | None = None
    questions: list[ClarifyingQuestion] = Field(default_factory=list)


class SynthesizerOutput(BaseModel):
    """Final report wrapper. The actual content lives in `report_markdown`."""

    model_config = ConfigDict(extra="ignore")

    report_markdown: str = Field(
        description="The full final report as markdown, including a Sources section."
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Notable cross-source contradictions surfaced during synthesis.",
    )
