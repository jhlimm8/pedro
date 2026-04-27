"""Thin async wrapper around OpenAI's Responses API.

Mirrors the mdb pattern (see [evals/framework/judge_runtime.py](../../mdb/evals/framework/judge_runtime.py)):
single `client.responses.parse(...)` call with `text_format=PydanticSchema`
plus the built-in `web_search_preview` tool. Returns the parsed Pydantic
instance plus token usage.

Designed for dependency injection: nodes accept an `LLMClient` (Protocol) and
tests pass a `FakeLLMClient` with canned responses. No network in unit tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from pedro.config import ReasoningEffort, Settings

T = TypeVar("T", bound=BaseModel)

WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search_preview"}


@dataclass
class ParsedResponse[T: BaseModel]:
    """Result of a structured-output LLM call."""

    parsed: T
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class LLMClient(Protocol):
    """The interface every node depends on. Real impl wraps OpenAI; tests use a fake."""

    async def parse(
        self,
        *,
        instructions: str,
        user_input: str,
        text_format: type[T],
        with_web_search: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int | None = None,
    ) -> ParsedResponse[T]: ...


class OpenAILLMClient:
    """Production LLM client backed by OpenAI's Responses API."""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_s,
            max_retries=2,
        )

    async def parse(
        self,
        *,
        instructions: str,
        user_input: str,
        text_format: type[T],
        with_web_search: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int | None = None,
    ) -> ParsedResponse[T]:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "input": user_input,
            "instructions": instructions,
            "text_format": text_format,
        }
        if with_web_search:
            kwargs["tools"] = [WEB_SEARCH_TOOL]
        effort = reasoning_effort or self._settings.reasoning_effort
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens

        start = time.perf_counter()
        response = await self._client.responses.parse(**kwargs)
        elapsed = time.perf_counter() - start

        if response.output_parsed is None:
            raise RuntimeError(
                f"OpenAI returned no parsed output for schema {text_format.__name__}. "
                f"Raw status: {getattr(response, 'status', 'unknown')}"
            )

        usage = response.usage
        return ParsedResponse(
            parsed=response.output_parsed,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            latency_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """In-memory fake for deterministic tests.

    Configure it by stacking responses in `.responses` (a list of Pydantic
    instances or callables that return one). Each `parse(...)` call pops the
    next response. If a callable is provided, it receives the kwargs dict so
    tests can assert on prompts.
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: list[Any] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def queue(self, response: Any) -> None:
        self.responses.append(response)

    async def parse(
        self,
        *,
        instructions: str,
        user_input: str,
        text_format: type[T],
        with_web_search: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int | None = None,
    ) -> ParsedResponse[T]:
        call = {
            "instructions": instructions,
            "user_input": user_input,
            "text_format": text_format,
            "with_web_search": with_web_search,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
        }
        self.calls.append(call)
        if not self.responses:
            raise AssertionError(
                f"FakeLLMClient: no queued response for {text_format.__name__}. "
                f"Stack a response with .queue() or pass via constructor. "
                f"Prior call_count={len(self.calls) - 1}."
            )
        nxt = self.responses.pop(0)
        if callable(nxt):
            nxt = nxt(call)
        if not isinstance(nxt, text_format):
            # Allow returning a dict that we coerce
            if isinstance(nxt, dict):
                nxt = text_format.model_validate(nxt)
            else:
                raise AssertionError(
                    f"FakeLLMClient: queued response of type {type(nxt).__name__} "
                    f"does not match expected {text_format.__name__}. "
                    f"call_index={len(self.calls) - 1}, remaining_queue_types="
                    f"{[type(r).__name__ for r in self.responses]}"
                )
        return ParsedResponse(parsed=nxt)
