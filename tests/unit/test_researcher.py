"""Researcher primitive tests.

Each test stacks canned ResearcherOutput responses on a FakeLLMClient and
verifies the primitive's coerce/emit behavior, prompt selection, and fan-out.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pedro.agents.deps import Deps
from pedro.agents.nodes.researcher import research_many, research_one
from pedro.agents.schemas import (
    Finding,
    ResearcherOutput,
    Source,
    SubQuestion,
)
from pedro.config import Settings
from pedro.llm.client import FakeLLMClient


def make_settings() -> Settings:
    return Settings(
        openai_api_key="",
        openai_base_url=None,
        model="gpt-test",
        reasoning_effort="low",
        scout_max_subquestions=5,
        deep_max_subquestions=8,
        request_timeout_s=180,
        trace_dir="traces",
    )


def make_finding(fid: str, sub_q_id: str = "", sources: list[Source] | None = None) -> Finding:
    return Finding(
        id=fid,
        headline=f"H {fid}",
        detail=f"D {fid}",
        sources=sources or [Source(url=f"https://example.com/{fid}", title=fid)],
        sub_question_id=sub_q_id,
    )


class CapturingDeps:
    def __init__(self, llm: FakeLLMClient) -> None:
        self.events: list[BaseModel] = []
        self.deps = Deps(
            llm=llm,
            settings=make_settings(),
            emit=self._emit,
        )

    async def _emit(self, event: BaseModel) -> None:
        self.events.append(event)


async def test_research_one_scout_phase_uses_scout_prompt():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("f1"), make_finding("f2")]),
    ])
    capt = CapturingDeps(fake)

    sq = SubQuestion(id="ssq1", question="What is X?")
    findings = await research_one(
        deps=capt.deps,
        phase="scout",
        overall_query="What is X overall?",
        interpreted_intent="",
        sub_question=sq,
    )

    assert len(findings) == 2
    assert all(f.sub_question_id == "ssq1" for f in findings), \
        "primitive must coerce sub_question_id from the request, even if model omits it"

    call = fake.calls[0]
    assert "Scout researcher" in call["instructions"]
    assert "<sub_question id=\"ssq1\">" in call["user_input"]
    assert "What is X?" in call["user_input"]
    assert call["with_web_search"] is True


async def test_research_one_deep_phase_uses_deep_prompt_with_motivating_findings():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("d1")]),
    ])
    capt = CapturingDeps(fake)

    sq = SubQuestion(
        id="sq1",
        question="Detailed question?",
        rationale="Because.",
        success_criteria="Two sources minimum.",
        depth="deep",
    )
    motivating = [make_finding("scout-1", "ssq1")]
    findings = await research_one(
        deps=capt.deps,
        phase="deep",
        overall_query="Q",
        interpreted_intent="The user wants a comparison.",
        sub_question=sq,
        motivating_findings=motivating,
    )

    assert len(findings) == 1
    call = fake.calls[0]
    assert "Deep researcher" in call["instructions"]
    assert "The user wants a comparison." in call["user_input"]
    assert "scout-1" in call["user_input"], \
        "deep prompt must inline motivating scout findings"
    assert "depth=\"deep\"" in call["user_input"]


async def test_research_one_emits_progress_and_finding_events():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("f1")]),
    ])
    capt = CapturingDeps(fake)

    sq = SubQuestion(id="ssq1", question="Q?")
    await research_one(
        deps=capt.deps,
        phase="scout",
        overall_query="Q?",
        interpreted_intent="",
        sub_question=sq,
    )

    types = [type(e).__name__ for e in capt.events]
    assert types[0] == "SubQuestionProgressEvent"
    assert any(t == "ScoutFindingEvent" for t in types)
    assert types[-1] == "SubQuestionProgressEvent"

    starts = [e for e in capt.events if type(e).__name__ == "SubQuestionProgressEvent"]
    assert getattr(starts[0], "status") == "started"
    assert getattr(starts[-1], "status") == "completed"


async def test_research_one_deep_phase_emits_source_found_per_source():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[
            make_finding("d1", sources=[
                Source(url="https://a.com", title="A"),
                Source(url="https://b.com", title="B"),
            ]),
        ]),
    ])
    capt = CapturingDeps(fake)

    sq = SubQuestion(id="sq1", question="?")
    await research_one(
        deps=capt.deps,
        phase="deep",
        overall_query="Q",
        interpreted_intent="I",
        sub_question=sq,
    )

    src_events = [e for e in capt.events if type(e).__name__ == "SourceFoundEvent"]
    assert len(src_events) == 2
    urls = {getattr(e, "source").url for e in src_events}
    assert urls == {"https://a.com", "https://b.com"}


async def test_research_many_fans_out_concurrently_and_aggregates():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("f1a"), make_finding("f1b")]),
        ResearcherOutput(findings=[make_finding("f2a")]),
        ResearcherOutput(findings=[make_finding("f3a"), make_finding("f3b"), make_finding("f3c")]),
    ])
    capt = CapturingDeps(fake)

    sub_qs = [
        SubQuestion(id="ssq1", question="A"),
        SubQuestion(id="ssq2", question="B"),
        SubQuestion(id="ssq3", question="C"),
    ]
    findings = await research_many(
        deps=capt.deps,
        phase="scout",
        overall_query="Q",
        interpreted_intent="",
        sub_questions=sub_qs,
    )

    assert len(findings) == 6
    by_sq = {f.sub_question_id for f in findings}
    assert by_sq == {"ssq1", "ssq2", "ssq3"}
    assert len(fake.calls) == 3


async def test_research_many_passes_motivating_findings_per_subquestion():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("d1")]),
        ResearcherOutput(findings=[make_finding("d2")]),
    ])
    capt = CapturingDeps(fake)

    sub_qs = [
        SubQuestion(id="sq1", question="A"),
        SubQuestion(id="sq2", question="B"),
    ]
    motivating = {
        "sq1": [make_finding("scout-A", "ssq1")],
        "sq2": [make_finding("scout-B", "ssq2")],
    }
    await research_many(
        deps=capt.deps,
        phase="deep",
        overall_query="Q",
        interpreted_intent="I",
        sub_questions=sub_qs,
        motivating_findings_by_sq=motivating,
    )

    bodies = [c["user_input"] for c in fake.calls]
    sq1_body = next(b for b in bodies if "sq1" in b)
    sq2_body = next(b for b in bodies if "sq2" in b)
    assert "scout-A" in sq1_body and "scout-B" not in sq1_body
    assert "scout-B" in sq2_body and "scout-A" not in sq2_body
