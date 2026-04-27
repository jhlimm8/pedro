"""Per-run dependency container threaded into LangGraph nodes via config.

Pattern: every node reads `deps = config["configurable"]["deps"]`. This keeps
LLM, settings, and the SSE event emitter out of the graph state proper (graph
state should be domain data only, not infra), and gives tests a clean injection
point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from pydantic import BaseModel

from pedro.config import Settings
from pedro.llm.client import LLMClient


EmitFn = Callable[[BaseModel], Awaitable[None]]


async def _noop_emit(event: BaseModel) -> None:  # noqa: ARG001
    return None


@dataclass
class Deps:
    """Runtime deps for graph nodes. Constructed once per session/run."""

    llm: LLMClient
    settings: Settings
    emit: EmitFn = field(default=_noop_emit)


def deps_from_config(config) -> Deps:  # accepts dict-like RunnableConfig
    """Extract Deps from a LangGraph RunnableConfig dict."""
    if not config:
        raise RuntimeError("Node was called without a RunnableConfig containing deps.")
    configurable = config.get("configurable") or {}
    deps = configurable.get("deps")
    if deps is None:
        raise RuntimeError(
            "Missing `deps` in config['configurable']. Build the graph with "
            "`graph.with_config({'configurable': {'deps': Deps(...)}})` or pass "
            "`config={'configurable': {'deps': Deps(...)}}` to invoke/astream."
        )
    return deps
