"""Cold + Plan+ planner node tests."""

from __future__ import annotations

from pydantic import BaseModel

from pedro.agents.deps import Deps
from pedro.agents.nodes.planner_cold import planner_cold_node
from pedro.agents.nodes.planner_plus import planner_plus_node
from pedro.agents.schemas import (
    ChatTurn,
    ClarifyingQuestion,
    Finding,
    PlannerOutput,
    ResearchPlan,
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


def make_plan() -> ResearchPlan:
    return ResearchPlan(
        interpreted_intent="Compare A and B.",
        scope_in=["since 2020"],
        sub_questions=[SubQuestion(id="sq1", question="What is A?")],
        expected_deliverable="A short comparison.",
    )


# --- Cold planner ---------------------------------------------------------


async def test_planner_cold_proposes_plan_with_no_web_search():
    fake = FakeLLMClient([make_plan()])
    deps, events = make_deps(fake)

    result = await planner_cold_node(
        {"user_query": "Compare A and B"}, config_for(deps)
    )
    assert result["proposed_plan"] is not None
    assert result["awaiting_clarification"] is False
    assert "(cold draft" in result["planner_thought"]

    assert fake.calls[0]["with_web_search"] is False, "cold planner is intentionally web-search-free"

    types = [type(e).__name__ for e in events]
    assert "PlanProposedEvent" in types
    # AwaitingApprovalEvent is emitted by Session._run_planning *after* the
    # graph pauses at approval_gate, not by the planner itself, to avoid a
    # race where consumers respond before the graph has paused.
    assert "AwaitingApprovalEvent" not in types


async def test_planner_cold_errors_without_query():
    fake = FakeLLMClient([])
    deps, _ = make_deps(fake)
    result = await planner_cold_node({}, config_for(deps))
    assert "error" in result and "user_query" in result["error"]


# --- Plan+ planner --------------------------------------------------------


def make_finding(fid: str, sub_q_id: str = "ssq1") -> Finding:
    return Finding(
        id=fid,
        headline=f"H {fid}",
        detail=f"D {fid}",
        sources=[Source(url=f"https://e.com/{fid}")],
        sub_question_id=sub_q_id,
    )


async def test_planner_plus_proposes_plan_when_findings_sufficient():
    fake_plan = ResearchPlan(
        interpreted_intent="...",
        sub_questions=[
            SubQuestion(id="sq1", question="?", motivating_findings=["f1"]),
        ],
        expected_deliverable="...",
    )
    fake = FakeLLMClient([
        PlannerOutput(kind="plan", thought="reasoning...", plan=fake_plan),
    ])
    deps, events = make_deps(fake)

    state = {
        "user_query": "Compare A and B",
        "scout_findings": [make_finding("f1"), make_finding("f2")],
        "chat_history": [],
    }
    result = await planner_plus_node(state, config_for(deps))

    assert result["awaiting_clarification"] is False
    assert result["proposed_plan"] is not None
    assert result["planner_thought"] == "reasoning..."

    types = [type(e).__name__ for e in events]
    assert "PlanProposedEvent" in types
    # See note on planner_cold: AwaitingApprovalEvent is emitted by Session.
    assert "AwaitingApprovalEvent" not in types
    # The thought-process narration must precede the plan event.
    asst_idx = types.index("AssistantMessageEvent")
    plan_idx = types.index("PlanProposedEvent")
    assert asst_idx < plan_idx

    call = fake.calls[0]
    assert "scout_findings" in call["user_input"] or "<scout_findings>" in call["user_input"]
    assert "f1" in call["user_input"], "scout finding ids must be inlined into the planner prompt"
    assert call["with_web_search"] is False


async def test_planner_plus_asks_clarifying_questions():
    fake = FakeLLMClient([
        PlannerOutput(
            kind="clarify",
            thought="Two distinct framings found; need direction.",
            questions=[ClarifyingQuestion(id="q1", question="Which framing?")],
        ),
    ])
    deps, events = make_deps(fake)

    state = {
        "user_query": "Tell me about X",
        "scout_findings": [make_finding("f1")],
        "chat_history": [ChatTurn(role="user", content="Tell me about X")],
    }
    result = await planner_plus_node(state, config_for(deps))

    assert result["awaiting_clarification"] is True
    assert result["proposed_plan"] is None
    # Questions are stashed in state for the clarify_gate to expose via
    # interrupt(); Session emits the SSE event after the graph pauses.
    pending = result["pending_clarifications"]
    assert len(pending) == 1
    assert pending[0].id == "q1"

    types = [type(e).__name__ for e in events]
    assert "ClarifyingQuestionsEvent" not in types
    assert "PlanProposedEvent" not in types
    assert "AwaitingApprovalEvent" not in types


async def test_planner_plus_errors_without_scout_findings():
    fake = FakeLLMClient([])
    deps, _ = make_deps(fake)
    state = {"user_query": "q", "scout_findings": []}
    result = await planner_plus_node(state, config_for(deps))
    assert "error" in result
    assert "scout_findings" in result["error"]


async def test_planner_plus_renders_chat_history_into_prompt():
    fake_plan = ResearchPlan(
        interpreted_intent="...",
        sub_questions=[SubQuestion(id="sq1", question="?")],
        expected_deliverable="...",
    )
    fake = FakeLLMClient([PlannerOutput(kind="plan", thought="t", plan=fake_plan)])
    deps, _ = make_deps(fake)

    state = {
        "user_query": "q",
        "scout_findings": [make_finding("f1")],
        "chat_history": [
            ChatTurn(role="user", content="MY-FIRST-MESSAGE"),
            ChatTurn(role="assistant", content="MY-RESPONSE"),
        ],
    }
    await planner_plus_node(state, config_for(deps))
    body = fake.calls[0]["user_input"]
    assert "MY-FIRST-MESSAGE" in body
    assert "MY-RESPONSE" in body
