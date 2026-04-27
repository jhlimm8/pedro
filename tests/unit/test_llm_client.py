"""Tests for the LLM client wrapper and FakeLLMClient."""

from __future__ import annotations

import pytest

from pedro.agents.schemas import ResearchPlan, ScoutPlanOutput
from pedro.llm.client import FakeLLMClient


async def test_fake_returns_queued_pydantic_instance():
    fake = FakeLLMClient([
        ScoutPlanOutput(interpretation="x", sub_questions=[]),
    ])
    result = await fake.parse(
        instructions="i",
        user_input="u",
        text_format=ScoutPlanOutput,
    )
    assert isinstance(result.parsed, ScoutPlanOutput)
    assert result.parsed.interpretation == "x"


async def test_fake_returns_queued_dict_coerced():
    fake = FakeLLMClient([
        {"interpretation": "y", "sub_questions": []},
    ])
    result = await fake.parse(
        instructions="i",
        user_input="u",
        text_format=ScoutPlanOutput,
    )
    assert isinstance(result.parsed, ScoutPlanOutput)
    assert result.parsed.interpretation == "y"


async def test_fake_records_calls():
    fake = FakeLLMClient([ScoutPlanOutput(interpretation="x", sub_questions=[])])
    await fake.parse(
        instructions="some-system",
        user_input="some-user",
        text_format=ScoutPlanOutput,
        reasoning_effort="medium",
    )
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["instructions"] == "some-system"
    assert call["user_input"] == "some-user"
    assert call["reasoning_effort"] == "medium"
    assert call["text_format"] is ScoutPlanOutput


async def test_fake_callable_responses_get_call_kwargs():
    captured: dict = {}

    def gen(call):
        captured.update(call)
        return ScoutPlanOutput(interpretation=f"got:{call['user_input']}", sub_questions=[])

    fake = FakeLLMClient([gen])
    result = await fake.parse(
        instructions="i",
        user_input="hello",
        text_format=ScoutPlanOutput,
    )
    assert result.parsed.interpretation == "got:hello"
    assert captured["instructions"] == "i"


async def test_fake_raises_on_empty_stack():
    fake = FakeLLMClient([])
    with pytest.raises(AssertionError, match="no queued response"):
        await fake.parse(
            instructions="i",
            user_input="u",
            text_format=ScoutPlanOutput,
        )


async def test_fake_raises_on_type_mismatch():
    fake = FakeLLMClient([
        ScoutPlanOutput(interpretation="x", sub_questions=[]),
    ])
    # Asking for a different format than queued
    with pytest.raises(AssertionError, match="does not match"):
        await fake.parse(
            instructions="i",
            user_input="u",
            text_format=ResearchPlan,
        )
