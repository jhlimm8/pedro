"""Schema round-trip and rendering tests."""

from __future__ import annotations

import json

from pedro.agents.schemas import (
    ChatTurn,
    ClarifyingQuestion,
    Finding,
    PlannerOutput,
    ResearchPlan,
    Source,
    SubQuestion,
)


def make_plan(with_findings: bool = False) -> ResearchPlan:
    motivating = ["f1", "f2"] if with_findings else []
    return ResearchPlan(
        interpreted_intent="Compare A and B for use in C.",
        scope_in=["recent papers", "industry reports"],
        scope_out=["pre-2020 work"],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="What is A?",
                rationale="Establish baseline.",
                motivating_findings=motivating,
                target_source_types=["academic", "wiki"],
                depth="medium",
                success_criteria="A short definition with at least 2 sources.",
            ),
            SubQuestion(
                id="sq2",
                question="What is B?",
                target_source_types=["industry"],
                depth="deep",
            ),
        ],
        expected_deliverable="A comparison table with citations.",
    )


def test_plan_round_trip():
    plan = make_plan(with_findings=True)
    js = plan.model_dump_json()
    rebuilt = ResearchPlan.model_validate_json(js)
    assert rebuilt == plan


def test_plan_to_markdown_contains_key_sections():
    md = make_plan(with_findings=True).to_markdown()
    assert "# Research Plan" in md
    assert "## Interpreted intent" in md
    assert "## Scope" in md
    assert "**In scope:**" in md
    assert "**Out of scope:**" in md
    assert "## Sub-questions" in md
    assert "### 1. What is A?" in md
    assert "### 2. What is B?" in md
    assert "## Expected deliverable" in md
    # Finding refs render
    assert "`f1`" in md and "`f2`" in md
    # Depth meta renders
    assert "depth: medium" in md
    assert "depth: deep" in md


def test_plan_markdown_handles_empty_lists():
    plan = ResearchPlan(
        interpreted_intent="x",
        sub_questions=[],
        expected_deliverable="y",
    )
    md = plan.to_markdown()
    assert "(none specified)" in md
    assert "_(no sub-questions)_" in md


def test_finding_model():
    f = Finding(
        id="f1",
        headline="Short take",
        detail="Long paragraph.",
        sources=[Source(url="https://example.com", title="Example")],
        sub_question_id="sq1",
    )
    parsed = Finding.model_validate_json(f.model_dump_json())
    assert parsed.sources[0].url == "https://example.com"
    assert parsed.sources[0].tier is None


def test_chat_turn_default_ts():
    t = ChatTurn(role="user", content="hi")
    assert t.ts is not None
    js = t.model_dump_json()
    assert "user" in js


def test_planner_output_kind_plan():
    out = PlannerOutput(
        kind="plan",
        thought="Reasoning...",
        plan=make_plan(),
        questions=[],
    )
    rebuilt = PlannerOutput.model_validate_json(out.model_dump_json())
    assert rebuilt.kind == "plan"
    assert rebuilt.plan is not None
    assert rebuilt.questions == []


def test_planner_output_kind_clarify():
    out = PlannerOutput(
        kind="clarify",
        thought="Need more info.",
        questions=[ClarifyingQuestion(id="q1", question="Which framing?")],
    )
    rebuilt = PlannerOutput.model_validate_json(out.model_dump_json())
    assert rebuilt.kind == "clarify"
    assert rebuilt.plan is None
    assert len(rebuilt.questions) == 1


def test_planner_output_json_schema_serializable():
    """Sanity check: schema dict is well-formed (this is what we hand to
    text_format= in responses.parse)."""
    schema = PlannerOutput.model_json_schema()
    assert "properties" in schema
    # Ensure it round-trips through json
    json.dumps(schema)
