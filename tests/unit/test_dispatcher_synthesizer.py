"""Dispatcher and synthesizer node tests."""

from __future__ import annotations

from pydantic import BaseModel

from pedro.agents.deps import Deps
from pedro.agents.nodes.dispatcher import dispatcher_node
from pedro.agents.nodes.synthesizer import synthesizer_node
from pedro.agents.schemas import (
    Finding,
    ResearchPlan,
    ResearcherOutput,
    Source,
    SubQuestion,
    SynthesizerOutput,
)
from pedro.config import Settings
from pedro.llm.client import FakeLLMClient


def make_settings() -> Settings:
    return Settings(
        openai_api_key="",
        openai_base_url=None,
        model="gpt-test",
        reasoning_effort="low",
        scout_max_subquestions=4,
        deep_max_subquestions=8,
        request_timeout_s=180,
        trace_dir="traces",
    )


def make_deps(llm: FakeLLMClient) -> tuple[Deps, list[BaseModel]]:
    events: list[BaseModel] = []

    async def emit(e: BaseModel) -> None:
        events.append(e)

    return Deps(llm=llm, settings=make_settings(), emit=emit), events


def config_for(deps: Deps) -> dict:
    return {"configurable": {"deps": deps}}


def make_finding(fid: str, sq_id: str = "ssq1", urls: list[str] | None = None) -> Finding:
    urls = urls or [f"https://e.com/{fid}"]
    return Finding(
        id=fid,
        headline=f"H {fid}",
        detail=f"D {fid}",
        sources=[Source(url=u, title=fid) for u in urls],
        sub_question_id=sq_id,
    )


# --- Dispatcher ----------------------------------------------------------


async def test_dispatcher_runs_one_call_per_plan_sub_question():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("d1", "sq1")]),
        ResearcherOutput(findings=[make_finding("d2", "sq2")]),
    ])
    deps, events = make_deps(fake)

    plan = ResearchPlan(
        interpreted_intent="...",
        sub_questions=[
            SubQuestion(id="sq1", question="Q1"),
            SubQuestion(id="sq2", question="Q2"),
        ],
        expected_deliverable="...",
    )
    state = {
        "user_query": "q",
        "approved_plan": plan,
        "scout_findings": [],
    }
    result = await dispatcher_node(state, config_for(deps))

    assert len(result["deep_findings"]) == 2
    assert {f.sub_question_id for f in result["deep_findings"]} == {"sq1", "sq2"}
    assert len(fake.calls) == 2
    assert all(c["with_web_search"] for c in fake.calls), "deep phase MUST use web search"


async def test_dispatcher_passes_motivating_findings_per_subquestion():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("d1", "sq1")]),
    ])
    deps, _ = make_deps(fake)

    plan = ResearchPlan(
        interpreted_intent="...",
        sub_questions=[
            SubQuestion(id="sq1", question="?", motivating_findings=["scout-f1"]),
        ],
        expected_deliverable="...",
    )
    state = {
        "approved_plan": plan,
        "scout_findings": [
            make_finding("scout-f1", "ssq-A"),
            make_finding("scout-f2", "ssq-B"),  # not referenced by plan
        ],
    }
    await dispatcher_node(state, config_for(deps))

    body = fake.calls[0]["user_input"]
    assert "scout-f1" in body, "dispatcher must inline motivating scout findings into deep prompt"
    assert "scout-f2" not in body, "non-referenced scout findings must NOT leak in"


async def test_dispatcher_dedupes_sources_across_findings():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[
            make_finding("d1", "sq1", urls=["https://x.com", "https://y.com"]),
            make_finding("d2", "sq1", urls=["https://x.com", "https://z.com"]),
        ]),
    ])
    deps, _ = make_deps(fake)

    plan = ResearchPlan(
        interpreted_intent="x",
        sub_questions=[SubQuestion(id="sq1", question="?")],
        expected_deliverable="x",
    )
    result = await dispatcher_node({"approved_plan": plan}, config_for(deps))
    urls = {s.url for s in result["deep_sources"]}
    assert urls == {"https://x.com", "https://y.com", "https://z.com"}


async def test_dispatcher_emits_research_started_event():
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("d1", "sq1")]),
    ])
    deps, events = make_deps(fake)

    plan = ResearchPlan(
        interpreted_intent="x",
        sub_questions=[SubQuestion(id="sq1", question="?")],
        expected_deliverable="x",
    )
    await dispatcher_node({"approved_plan": plan}, config_for(deps))
    types = [type(e).__name__ for e in events]
    assert "ResearchStartedEvent" in types


async def test_dispatcher_errors_without_plan_or_subquestions():
    fake = FakeLLMClient([])
    deps, _ = make_deps(fake)

    r1 = await dispatcher_node({}, config_for(deps))
    assert "error" in r1 and "approved_plan" in r1["error"]

    plan = ResearchPlan(interpreted_intent="x", sub_questions=[], expected_deliverable="x")
    r2 = await dispatcher_node({"approved_plan": plan}, config_for(deps))
    assert "error" in r2 and "sub-questions" in r2["error"]


# --- Synthesizer ---------------------------------------------------------


async def test_synthesizer_renders_findings_into_prompt_and_returns_report():
    fake = FakeLLMClient([
        SynthesizerOutput(
            report_markdown="# Report\n\nTL;DR.\n\n## Sources\n1. X — https://x.com",
            contradictions=["X says A; Y says B"],
        ),
    ])
    deps, events = make_deps(fake)

    plan = ResearchPlan(
        interpreted_intent="Compare A and B.",
        sub_questions=[
            SubQuestion(id="sq1", question="What is A?", success_criteria="2 sources"),
        ],
        expected_deliverable="Comparison.",
    )
    deep_findings = [
        make_finding("d1", "sq1", urls=["https://x.com"]),
    ]

    result = await synthesizer_node(
        {"approved_plan": plan, "deep_findings": deep_findings},
        config_for(deps),
    )
    assert result["report_markdown"].startswith("# Report")
    assert result["contradictions"] == ["X says A; Y says B"]

    body = fake.calls[0]["user_input"]
    assert "Compare A and B." in body, "interpreted_intent must be in synthesizer prompt"
    assert "id=sq1" in body
    assert "[d1]" in body
    assert "https://x.com" in body
    assert fake.calls[0]["with_web_search"] is False, "synthesizer must not re-search"

    types = [type(e).__name__ for e in events]
    assert types == ["SynthesisStartedEvent", "FinalReportEvent"]


async def test_synthesizer_errors_without_findings_or_plan():
    fake = FakeLLMClient([])
    deps, _ = make_deps(fake)

    r1 = await synthesizer_node({}, config_for(deps))
    assert "error" in r1 and "approved_plan" in r1["error"]

    plan = ResearchPlan(
        interpreted_intent="x",
        sub_questions=[SubQuestion(id="sq1", question="?")],
        expected_deliverable="x",
    )
    r2 = await synthesizer_node({"approved_plan": plan, "deep_findings": []}, config_for(deps))
    assert "error" in r2 and "deep_findings" in r2["error"]
