"""Scout node tests."""

from __future__ import annotations

from pydantic import BaseModel

from pedro.agents.deps import Deps
from pedro.agents.nodes.scout import scout_node
from pedro.agents.schemas import (
    Finding,
    ResearcherOutput,
    ScoutPlanOutput,
    ScoutSubQuestion,
    Source,
)
from pedro.config import Settings
from pedro.llm.client import FakeLLMClient


def make_settings(**overrides) -> Settings:
    base = dict(
        openai_api_key="",
        openai_base_url=None,
        model="gpt-test",
        reasoning_effort="low",
        scout_max_subquestions=3,
        deep_max_subquestions=8,
        request_timeout_s=180,
        trace_dir="traces",
    )
    base.update(overrides)
    return Settings(**base)


def make_deps(llm: FakeLLMClient) -> tuple[Deps, list[BaseModel]]:
    events: list[BaseModel] = []

    async def emit(e: BaseModel) -> None:
        events.append(e)

    return Deps(llm=llm, settings=make_settings(), emit=emit), events


def make_finding(fid: str, sub_q_id: str = "") -> Finding:
    return Finding(
        id=fid,
        headline=f"H {fid}",
        detail=f"D {fid}",
        sources=[Source(url=f"https://x.com/{fid}")],
        sub_question_id=sub_q_id,
    )


def config_for(deps: Deps) -> dict:
    return {"configurable": {"deps": deps}}


async def test_scout_node_decomposes_then_fans_out():
    fake = FakeLLMClient([
        ScoutPlanOutput(
            interpretation="The user wants to understand the landscape of X.",
            sub_questions=[
                ScoutSubQuestion(id="ssq1", question="What framings exist?"),
                ScoutSubQuestion(id="ssq2", question="Who are key actors?"),
            ],
        ),
        ResearcherOutput(findings=[make_finding("f1"), make_finding("f2")]),
        ResearcherOutput(findings=[make_finding("f3")]),
    ])
    deps, events = make_deps(fake)

    state = {"user_query": "Tell me about X.", "mode": "plan_plus"}
    result = await scout_node(state, config_for(deps))

    assert "error" not in result
    assert len(result["scout_sub_questions"]) == 2
    assert {sq.id for sq in result["scout_sub_questions"]} == {"ssq1", "ssq2"}
    assert all(sq.depth == "shallow" for sq in result["scout_sub_questions"])
    assert len(result["scout_findings"]) == 3
    assert {f.sub_question_id for f in result["scout_findings"]} == {"ssq1", "ssq2"}

    # First call had no web search (planner step)
    assert fake.calls[0]["with_web_search"] is False
    # Subsequent calls had web search (researcher step)
    assert fake.calls[1]["with_web_search"] is True
    assert fake.calls[2]["with_web_search"] is True


async def test_scout_node_emits_lifecycle_events():
    fake = FakeLLMClient([
        ScoutPlanOutput(
            interpretation="...",
            sub_questions=[ScoutSubQuestion(id="ssq1", question="?")],
        ),
        ResearcherOutput(findings=[make_finding("f1")]),
    ])
    deps, events = make_deps(fake)

    await scout_node({"user_query": "q"}, config_for(deps))

    types = [type(e).__name__ for e in events]
    assert "AssistantMessageEvent" in types  # interpretation narration
    assert "ScoutStartedEvent" in types
    assert "ScoutSubQuestionEvent" in types
    assert "ScoutFindingEvent" in types
    assert "ScoutCompleteEvent" in types
    # Started must come before the per-finding events; Complete must come last.
    assert types.index("ScoutStartedEvent") < types.index("ScoutFindingEvent")
    assert types[-1] == "ScoutCompleteEvent"


async def test_scout_node_returns_error_when_planner_yields_no_subquestions():
    fake = FakeLLMClient([
        ScoutPlanOutput(interpretation="x", sub_questions=[]),
    ])
    deps, _ = make_deps(fake)
    result = await scout_node({"user_query": "q"}, config_for(deps))
    assert "error" in result
    assert "no sub-questions" in result["error"]


async def test_scout_node_errors_on_missing_query():
    fake = FakeLLMClient([])
    deps, _ = make_deps(fake)
    result = await scout_node({}, config_for(deps))
    assert "error" in result
    assert "user_query" in result["error"]
