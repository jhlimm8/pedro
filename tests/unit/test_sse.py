"""SSE event protocol tests."""

from __future__ import annotations

import json

from pedro.agents.schemas import ClarifyingQuestion, Finding, ResearchPlan, Source
from pedro.api.sse import (
    AssistantMessageEvent,
    ClarifyingQuestionsEvent,
    DoneEvent,
    ErrorEvent,
    FinalReportEvent,
    ModeSetEvent,
    PlanProposedEvent,
    ScoutFindingEvent,
    SourceFoundEvent,
    SubQuestionProgressEvent,
    serialize_sse,
)


def test_mode_set_event_round_trip():
    e = ModeSetEvent(mode="plan_plus")
    payload = serialize_sse(e)
    assert payload["event"] == "mode_set"
    parsed = json.loads(payload["data"])
    assert parsed["mode"] == "plan_plus"
    assert parsed["type"] == "mode_set"
    assert "ts" in parsed


def test_plan_proposed_event_carries_plan_and_markdown():
    plan = ResearchPlan(interpreted_intent="x", expected_deliverable="y")
    e = PlanProposedEvent(plan=plan, plan_markdown=plan.to_markdown())
    payload = serialize_sse(e)
    parsed = json.loads(payload["data"])
    assert parsed["type"] == "plan_proposed"
    assert "plan" in parsed and "plan_markdown" in parsed
    assert "# Research Plan" in parsed["plan_markdown"]


def test_assistant_message_event():
    e = AssistantMessageEvent(content="**hi**")
    payload = serialize_sse(e)
    assert payload["event"] == "assistant_message"
    assert json.loads(payload["data"])["content"] == "**hi**"


def test_clarifying_questions_event():
    e = ClarifyingQuestionsEvent(
        questions=[ClarifyingQuestion(id="q1", question="?")]
    )
    payload = serialize_sse(e)
    assert json.loads(payload["data"])["questions"][0]["id"] == "q1"


def test_scout_finding_event_carries_finding():
    f = Finding(
        id="f1",
        headline="x",
        detail="y",
        sources=[Source(url="https://e.com")],
        sub_question_id="ssq1",
    )
    e = ScoutFindingEvent(finding=f)
    parsed = json.loads(serialize_sse(e)["data"])
    assert parsed["finding"]["id"] == "f1"
    assert parsed["finding"]["sources"][0]["url"] == "https://e.com"


def test_source_found_event():
    e = SourceFoundEvent(
        sub_question_id="sq1",
        source=Source(url="https://e.com", title="t"),
    )
    parsed = json.loads(serialize_sse(e)["data"])
    assert parsed["sub_question_id"] == "sq1"
    assert parsed["source"]["title"] == "t"


def test_subquestion_progress_event_status_literal():
    e = SubQuestionProgressEvent(sub_question_id="sq1", status="started")
    parsed = json.loads(serialize_sse(e)["data"])
    assert parsed["status"] == "started"


def test_final_report_event():
    e = FinalReportEvent(report_markdown="# Report\n", contradictions=["A vs B"])
    parsed = json.loads(serialize_sse(e)["data"])
    assert parsed["report_markdown"].startswith("# Report")
    assert parsed["contradictions"] == ["A vs B"]


def test_error_and_done_events():
    err = ErrorEvent(message="boom")
    done = DoneEvent()
    assert serialize_sse(err)["event"] == "error"
    assert serialize_sse(done)["event"] == "done"
