"""End-to-end graph tests with FakeLLMClient.

These exercise the full state-machine: scout -> planner_plus -> clarify
turn -> planner_plus -> approval (edit) -> planner_plus -> approval (approve)
-> ... and a complete Plan+ -> Research handoff.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from pedro.agents.deps import Deps
from pedro.agents.graph import (
    build_plan_graph,
    build_plus_graph,
    build_research_graph,
)
from pedro.agents.schemas import (
    ClarifyingQuestion,
    Finding,
    PlannerOutput,
    ResearchPlan,
    ResearcherOutput,
    ScoutPlanOutput,
    ScoutSubQuestion,
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
        scout_max_subquestions=3,
        deep_max_subquestions=8,
        request_timeout_s=180,
        trace_dir="traces",
    )


def collecting_emit() -> tuple[Any, list[BaseModel]]:
    events: list[BaseModel] = []

    async def emit(e: BaseModel) -> None:
        events.append(e)

    return emit, events


def config_for(deps: Deps, thread_id: str = "t1") -> dict:
    return {"configurable": {"deps": deps, "thread_id": thread_id}}


def make_finding(fid: str, sq_id: str = "ssq1", urls: list[str] | None = None) -> Finding:
    urls = urls or [f"https://e.com/{fid}"]
    return Finding(
        id=fid,
        headline=f"H {fid}",
        detail=f"D {fid}",
        sources=[Source(url=u, title=fid) for u in urls],
        sub_question_id=sq_id,
    )


# ---------------------------------------------------------------------------
# Plan mode (cold)
# ---------------------------------------------------------------------------


async def test_plan_graph_proposes_plan_then_pauses_for_approval():
    """Cold planner produces plan, approval_gate pauses, resume with approve."""
    plan = ResearchPlan(
        interpreted_intent="Cold draft.",
        sub_questions=[SubQuestion(id="sq1", question="Q?")],
        expected_deliverable="Output.",
    )
    fake = FakeLLMClient([plan])
    emit, events = collecting_emit()
    deps = Deps(llm=fake, settings=make_settings(), emit=emit)
    saver = InMemorySaver()
    graph = build_plan_graph(checkpointer=saver)

    result = await graph.ainvoke(
        {"user_query": "Compare A and B", "mode": "plan"},
        config=config_for(deps, "t-plan-1"),
    )

    # Graph paused at approval_gate -> __interrupt__ in result
    assert "__interrupt__" in result
    assert result["proposed_plan"] is not None
    assert result.get("approved_plan") is None

    types = [type(e).__name__ for e in events]
    assert "PlanProposedEvent" in types
    # AwaitingApprovalEvent is emitted by Session._run_planning post-pause,
    # not by the planner node, to avoid a checkpoint race with the SSE
    # consumer; it must NOT appear in events from the graph itself.
    assert "AwaitingApprovalEvent" not in types

    # Resume with approve
    resumed = await graph.ainvoke(
        Command(resume={"action": "approve"}),
        config=config_for(deps, "t-plan-1"),
    )
    assert resumed.get("approved_plan") is not None
    types_after = [type(e).__name__ for e in events]
    assert "PlanApprovedEvent" in types_after
    assert "ModeLockedEvent" in types_after


async def test_plan_graph_edit_loops_back_to_planner():
    """When approval_gate resumes with action=edit, planner is re-run."""
    plan_v1 = ResearchPlan(
        interpreted_intent="V1.",
        sub_questions=[SubQuestion(id="sq1", question="Q1?")],
        expected_deliverable="D",
    )
    plan_v2 = ResearchPlan(
        interpreted_intent="V2 — incorporated edit.",
        sub_questions=[SubQuestion(id="sq1", question="Q1-edited?")],
        expected_deliverable="D",
    )
    fake = FakeLLMClient([plan_v1, plan_v2])
    emit, _ = collecting_emit()
    deps = Deps(llm=fake, settings=make_settings(), emit=emit)
    saver = InMemorySaver()
    graph = build_plan_graph(checkpointer=saver)

    cfg = config_for(deps, "t-plan-edit")
    await graph.ainvoke({"user_query": "q", "mode": "plan"}, config=cfg)
    # Edit
    after_edit = await graph.ainvoke(
        Command(resume={"action": "edit", "edits": "Make it more specific."}),
        config=cfg,
    )
    # After edit, graph should re-run planner and pause again at approval
    assert "__interrupt__" in after_edit
    assert len(fake.calls) == 2, "edit should have triggered a 2nd planner call"
    # Approve v2
    final = await graph.ainvoke(Command(resume={"action": "approve"}), config=cfg)
    assert final.get("approved_plan") is not None
    assert final["approved_plan"].interpreted_intent == "V2 — incorporated edit."


# ---------------------------------------------------------------------------
# Plan+ mode
# ---------------------------------------------------------------------------


async def test_plus_graph_full_flow_with_clarification_then_approval():
    """Scout -> planner asks clarify -> user replies -> planner proposes plan -> approve."""
    scout_plan = ScoutPlanOutput(
        interpretation="Mapping the field of X.",
        sub_questions=[
            ScoutSubQuestion(id="ssq1", question="What framings exist?"),
            ScoutSubQuestion(id="ssq2", question="Who are key actors?"),
        ],
    )
    scout_findings_call_1 = ResearcherOutput(
        findings=[make_finding("f1", "ssq1"), make_finding("f2", "ssq1")]
    )
    scout_findings_call_2 = ResearcherOutput(findings=[make_finding("f3", "ssq2")])

    planner_clarify = PlannerOutput(
        kind="clarify",
        thought="Two distinct framings emerged.",
        questions=[ClarifyingQuestion(id="q1", question="Which framing matters more?")],
    )
    plan_after_clarify = ResearchPlan(
        interpreted_intent="Focus on framing-A based on user reply.",
        sub_questions=[
            SubQuestion(id="sq1", question="What does framing A imply?", motivating_findings=["f1"]),
        ],
        expected_deliverable="An A-focused report.",
    )
    planner_plan = PlannerOutput(kind="plan", thought="reasoning", plan=plan_after_clarify)

    fake = FakeLLMClient([
        scout_plan,
        scout_findings_call_1,
        scout_findings_call_2,
        planner_clarify,
        planner_plan,
    ])
    emit, events = collecting_emit()
    deps = Deps(llm=fake, settings=make_settings(), emit=emit)
    saver = InMemorySaver()
    graph = build_plus_graph(checkpointer=saver)
    cfg = config_for(deps, "t-plus-1")

    # Initial run: scout + planner_plus -> clarify pause
    r1 = await graph.ainvoke(
        {"user_query": "Tell me about X", "mode": "plan_plus"},
        config=cfg,
    )
    assert "__interrupt__" in r1
    assert r1.get("awaiting_clarification") is True
    types1 = [type(e).__name__ for e in events]
    assert "ScoutCompleteEvent" in types1
    # ClarifyingQuestionsEvent is emitted by Session._run_planning after the
    # graph pauses at clarify_gate. The questions live in the interrupt's
    # value here; we verify them via state instead.
    assert "ClarifyingQuestionsEvent" not in types1
    assert "PlanProposedEvent" not in types1
    pending = r1.get("pending_clarifications") or []
    assert any(q.id == "q1" for q in pending)

    # Resume with clarification reply -> planner re-runs, proposes plan, approval pause
    r2 = await graph.ainvoke(
        Command(resume={"text": "Framing A is what I care about."}),
        config=cfg,
    )
    assert "__interrupt__" in r2
    assert r2.get("proposed_plan") is not None
    types2 = [type(e).__name__ for e in events]
    assert "PlanProposedEvent" in types2
    # AwaitingApprovalEvent is emitted by Session, not by the graph.
    assert "AwaitingApprovalEvent" not in types2

    # Resume with approve
    r3 = await graph.ainvoke(Command(resume={"action": "approve"}), config=cfg)
    assert r3.get("approved_plan") is not None
    assert r3["approved_plan"].interpreted_intent == "Focus on framing-A based on user reply."
    types3 = [type(e).__name__ for e in events]
    assert "PlanApprovedEvent" in types3
    assert "ModeLockedEvent" in types3

    # Final state should preserve scout_findings AND approved_plan (handoff to Research)
    assert len(r3.get("scout_findings", [])) == 3


async def test_plus_graph_scout_planner_no_clarification_then_approve():
    """Happy path: scout produces enough context for planner to propose immediately."""
    scout_plan = ScoutPlanOutput(
        interpretation="OK.",
        sub_questions=[ScoutSubQuestion(id="ssq1", question="Broad?")],
    )
    plan = ResearchPlan(
        interpreted_intent="Direct plan.",
        sub_questions=[SubQuestion(id="sq1", question="Q?", motivating_findings=["f1"])],
        expected_deliverable="D.",
    )
    fake = FakeLLMClient([
        scout_plan,
        ResearcherOutput(findings=[make_finding("f1", "ssq1")]),
        PlannerOutput(kind="plan", thought="t", plan=plan),
    ])
    emit, _ = collecting_emit()
    deps = Deps(llm=fake, settings=make_settings(), emit=emit)
    saver = InMemorySaver()
    graph = build_plus_graph(checkpointer=saver)
    cfg = config_for(deps, "t-plus-2")

    r1 = await graph.ainvoke({"user_query": "q", "mode": "plan_plus"}, config=cfg)
    assert r1.get("proposed_plan") is not None
    assert r1.get("awaiting_clarification") is False
    r2 = await graph.ainvoke(Command(resume={"action": "approve"}), config=cfg)
    assert r2.get("approved_plan") is not None


# ---------------------------------------------------------------------------
# Research mode + headline Plan+ -> Research handoff
# ---------------------------------------------------------------------------


async def test_research_graph_runs_dispatcher_then_synthesizer():
    plan = ResearchPlan(
        interpreted_intent="Test.",
        sub_questions=[
            SubQuestion(id="sq1", question="A?"),
            SubQuestion(id="sq2", question="B?"),
        ],
        expected_deliverable="Comparison.",
    )
    fake = FakeLLMClient([
        ResearcherOutput(findings=[make_finding("d1", "sq1")]),
        ResearcherOutput(findings=[make_finding("d2", "sq2")]),
        SynthesizerOutput(
            report_markdown="# Final\nTL;DR.\n\n## Sources\n1. d1 — https://e.com/d1",
            contradictions=[],
        ),
    ])
    emit, events = collecting_emit()
    deps = Deps(llm=fake, settings=make_settings(), emit=emit)
    graph = build_research_graph()

    result = await graph.ainvoke(
        {"approved_plan": plan, "user_query": "q"},
        config=config_for(deps, "t-research-1"),
    )
    assert result["report_markdown"].startswith("# Final")
    assert len(result["deep_findings"]) == 2
    types = [type(e).__name__ for e in events]
    assert "ResearchStartedEvent" in types
    assert "FinalReportEvent" in types
    assert types.index("ResearchStartedEvent") < types.index("FinalReportEvent")


async def test_full_plus_to_research_handoff():
    """The headline integration test: Plan+ approves a plan, then the Research
    graph runs against the same in-memory state and produces a final report."""
    scout_plan = ScoutPlanOutput(
        interpretation="...",
        sub_questions=[ScoutSubQuestion(id="ssq1", question="?")],
    )
    plan = ResearchPlan(
        interpreted_intent="X",
        sub_questions=[SubQuestion(id="sq1", question="Q?", motivating_findings=["f1"])],
        expected_deliverable="D",
    )
    fake = FakeLLMClient([
        scout_plan,
        ResearcherOutput(findings=[make_finding("f1", "ssq1")]),
        PlannerOutput(kind="plan", thought="t", plan=plan),
        # research phase calls
        ResearcherOutput(findings=[make_finding("d1", "sq1")]),
        SynthesizerOutput(report_markdown="# Done\n", contradictions=[]),
    ])
    emit, events = collecting_emit()
    deps = Deps(llm=fake, settings=make_settings(), emit=emit)
    saver = InMemorySaver()
    plus = build_plus_graph(checkpointer=saver)
    research = build_research_graph()
    cfg = config_for(deps, "t-handoff")

    await plus.ainvoke({"user_query": "q", "mode": "plan_plus"}, config=cfg)
    plus_final = await plus.ainvoke(Command(resume={"action": "approve"}), config=cfg)

    # Hand off scout_findings + approved_plan to Research graph
    research_input = {
        "user_query": plus_final.get("user_query", "q"),
        "approved_plan": plus_final["approved_plan"],
        "scout_findings": plus_final.get("scout_findings", []),
    }
    research_final = await research.ainvoke(
        research_input, config=config_for(deps, "t-handoff-r")
    )
    assert research_final["report_markdown"].startswith("# Done")
    assert len(research_final["deep_findings"]) == 1
