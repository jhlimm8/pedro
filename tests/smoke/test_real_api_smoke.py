"""Env-gated end-to-end smoke test against the real OpenAI Responses API.

This test is intentionally NOT run by default. It exercises the full
Plan+ -> Research pipeline against a small, low-cost real query so we can
catch:
  - prompt regressions (the structured-output schemas still parse)
  - tool-call wiring (web_search_preview returns sources)
  - dispatcher fan-out + synthesizer happy path against real artifacts

Enable with:

    export OPENAI_API_KEY=sk-...
    export PEDRO_RUN_SMOKE=1
    pytest tests/smoke -s

Skipped otherwise so CI stays free + deterministic.
"""

from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from pedro.agents.deps import Deps
from pedro.agents.graph import build_plus_graph, build_research_graph
from pedro.config import Settings
from pedro.llm.client import OpenAILLMClient

SHOULD_RUN = os.getenv("PEDRO_RUN_SMOKE") == "1" and bool(os.getenv("OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not SHOULD_RUN,
    reason="set PEDRO_RUN_SMOKE=1 + OPENAI_API_KEY to enable real-API smoke",
)


@pytest.mark.asyncio
async def test_real_api_plus_to_research_smoke():
    """Run a small Plan+ -> Research on a benign query against the real API.

    We pick a query that should be answerable with a handful of web sources
    and that doesn't risk safety-classifier refusals: comparison of two
    publicly-documented technologies. The assertions are intentionally weak
    because real LLM output is stochastic; we just verify the *shape* of
    each phase's output.
    """
    settings = Settings.from_env()
    # Cap blast radius for cost.
    settings = Settings(
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
        model=settings.model,
        reasoning_effort="low",
        scout_max_subquestions=2,
        deep_max_subquestions=2,
        request_timeout_s=120,
        trace_dir=settings.trace_dir,
    )
    llm = OpenAILLMClient(settings)

    events: list[BaseModel] = []

    async def emit(e: BaseModel) -> None:
        events.append(e)

    deps = Deps(llm=llm, settings=settings, emit=emit)
    saver = InMemorySaver()
    plus = build_plus_graph(checkpointer=saver)
    cfg = {"configurable": {"deps": deps, "thread_id": "smoke-plus"}}

    # Use a query that's stable / has clear coverage online.
    query = "Compare Postgres logical replication vs physical replication."

    r1 = await plus.ainvoke(
        {"user_query": query, "mode": "plan_plus"}, config=cfg
    )
    assert "__interrupt__" in r1, "Plan+ should pause at clarify or approval"

    # If the planner asked clarifying questions, give a generic reply
    # ("optimize for the comparison-as-stated") and re-resume. Otherwise
    # we're already at approval.
    if r1.get("awaiting_clarification"):
        r1 = await plus.ainvoke(
            Command(resume={"text": "Stay with the comparison as I asked it."}),
            config=cfg,
        )
        assert "__interrupt__" in r1

    # Should now be at approval gate with a proposed plan.
    assert r1.get("proposed_plan") is not None
    plan = r1["proposed_plan"]
    assert plan.interpreted_intent
    assert len(plan.sub_questions) >= 1

    r2 = await plus.ainvoke(Command(resume={"action": "approve"}), config=cfg)
    approved = r2.get("approved_plan")
    assert approved is not None

    # Hand off to research graph.
    research = build_research_graph()
    r3 = await research.ainvoke(
        {
            "user_query": query,
            "approved_plan": approved,
            "scout_findings": r2.get("scout_findings", []),
        },
        config={"configurable": {"deps": deps, "thread_id": "smoke-research"}},
    )

    report = r3.get("report_markdown") or ""
    findings = r3.get("deep_findings") or []
    assert report, "synthesizer must produce a non-empty report"
    assert len(findings) >= 1, "deep researcher must produce at least one finding"
    # Every finding should have at least one source — that's the whole point
    # of using the web_search tool.
    assert all(len(f.sources) >= 1 for f in findings), (
        "every finding should cite at least one source"
    )
    # We routinely see at least one event of each major lifecycle type.
    types = [type(e).__name__ for e in events]
    for required in (
        "ScoutStartedEvent",
        "ScoutCompleteEvent",
        "PlanProposedEvent",
        "PlanApprovedEvent",
        "ModeLockedEvent",
        "ResearchStartedEvent",
        "FinalReportEvent",
    ):
        assert required in types, f"missing required SSE event type: {required}"
