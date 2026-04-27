"""Session manager: per-session graph state, SSE event queue, and run control.

A session represents one chat conversation. The mode-lock semantics live here:

  - While `locked=False`, the user can switch between Plan and Plan+ modes by
    starting a new session run; the chosen mode is run with a checkpointed
    LangGraph thread, and clarify/approval interrupts are resumed via
    `respond()`.
  - When the planning graph returns with `approved_plan` set, we set
    `locked=True`, emit ModeLockedEvent, and immediately run the Research
    graph using the same Session.deps.emit, streaming the deep-research
    progress and final report onto the same event queue.

Runs are dispatched as background asyncio tasks; the SSE stream endpoint
just consumes the per-session queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from pedro.agents.deps import Deps
from pedro.agents.graph import (
    build_plan_graph,
    build_plus_graph,
    build_research_graph,
)
from pedro.agents.schemas import ClarifyingQuestion, Finding, Mode, ResearchPlan
from pedro.api.sse import (
    AwaitingApprovalEvent,
    ClarifyingQuestionsEvent,
    DoneEvent,
    ErrorEvent,
    ModeSetEvent,
)
from pedro.config import Settings
from pedro.llm.client import LLMClient

log = logging.getLogger(__name__)


class Session:
    """One in-memory chat session."""

    def __init__(
        self,
        session_id: str,
        llm: LLMClient,
        settings: Settings,
        trace_dir: Path | None = None,
    ) -> None:
        self.id = session_id
        self.settings = settings
        self.queue: asyncio.Queue[BaseModel | None] = asyncio.Queue()
        self._seq = 0

        self.locked: bool = False
        self.mode: Mode | None = None
        self.user_query: str = ""
        self.approved_plan: ResearchPlan | None = None
        self.scout_findings: list[Finding] = []

        self._planning_thread = f"plan-{session_id}"
        self._research_thread = f"research-{session_id}"
        self._checkpointer = InMemorySaver()
        self._tasks: list[asyncio.Task] = []
        self._trace_path: Path | None = None
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_path = trace_dir / f"{session_id}.jsonl"

        self.deps = Deps(llm=llm, settings=settings, emit=self._emit)

    # -- emit / stream ------------------------------------------------------

    async def _emit(self, event: BaseModel) -> None:
        self._seq += 1
        if hasattr(event, "seq"):
            try:
                object.__setattr__(event, "seq", self._seq)
            except Exception:
                # Fallback: some pydantic models forbid mutation; skip.
                pass
        # Cooperative yield: ensure the SSE consumer task gets a chance to
        # drain even when producer awaits don't naturally yield (e.g. unbounded
        # asyncio.Queue.put returns immediately on an empty queue). Without
        # this, a fast LLM (or the FakeLLMClient) can starve the consumer.
        await asyncio.sleep(0)
        if self._trace_path is not None:
            try:
                line = json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        **event.model_dump(mode="json"),
                    },
                    default=str,
                )
                with self._trace_path.open("a") as f:
                    f.write(line + "\n")
            except Exception:
                # Tracing must never break the run.
                log.exception("trace write failed")
        await self.queue.put(event)

    async def stream_events(self) -> AsyncIterator[BaseModel]:
        """Async generator over events for SSE. Closed when DoneEvent is queued."""
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item
            if isinstance(item, DoneEvent):
                return

    # -- run control --------------------------------------------------------

    async def start(self, mode: Mode, user_query: str) -> None:
        if self.locked and mode != "research":
            raise RuntimeError(
                f"Session {self.id} is locked; cannot start mode={mode}."
            )
        if mode == "research" and not self.locked:
            raise RuntimeError(
                "Cannot start research mode before a plan is approved."
            )
        self.mode = mode
        self.user_query = user_query
        await self._emit(ModeSetEvent(mode=mode))

        if mode == "research":
            self._spawn(self._run_research())
            return

        if mode == "plan":
            graph = build_plan_graph(checkpointer=self._checkpointer)
        elif mode == "plan_plus":
            graph = build_plus_graph(checkpointer=self._checkpointer)
        else:  # pragma: no cover - guarded by Literal
            raise RuntimeError(f"Unknown mode: {mode}")

        cfg: RunnableConfig = {
            "configurable": {"deps": self.deps, "thread_id": self._planning_thread}
        }
        self._spawn(self._run_planning(graph, cfg, initial_input={"user_query": user_query, "mode": mode}))

    async def respond(self, payload: dict) -> None:
        """Resume the paused planning graph with the user's response."""
        if self.locked:
            raise RuntimeError("Session is locked; cannot respond to planning.")
        if self.mode not in ("plan", "plan_plus"):
            raise RuntimeError("Session has no active planning run to respond to.")
        if self.mode == "plan":
            graph = build_plan_graph(checkpointer=self._checkpointer)
        else:
            graph = build_plus_graph(checkpointer=self._checkpointer)
        cfg: RunnableConfig = {
            "configurable": {"deps": self.deps, "thread_id": self._planning_thread}
        }
        self._spawn(self._run_planning(graph, cfg, initial_input=Command(resume=payload)))

    # -- internals ----------------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.append(task)

    async def _run_planning(self, graph, cfg, *, initial_input) -> None:
        try:
            result = await graph.ainvoke(initial_input, config=cfg)
            if result.get("approved_plan") is not None:
                self.approved_plan = result["approved_plan"]
                self.scout_findings = result.get("scout_findings", []) or []
                self.locked = True
                # Auto-handoff: run Research on the same session, same emitter.
                await self._run_research()
                return
            # Paused at an interrupt: emit the matching awaiting event NOW,
            # *after* the graph has actually paused & checkpointed. Emitting
            # this from inside a planner node would race with the gate's
            # interrupt(): a fast consumer could call respond() before the
            # graph has paused, causing the next ainvoke to start from scratch
            # instead of resuming.
            await self._emit_awaiting_for_interrupt(result.get("__interrupt__"))
        except Exception as e:
            log.exception("planning run failed")
            await self._emit(ErrorEvent(message=f"planning failed: {e}"))
            await self.queue.put(None)

    async def _emit_awaiting_for_interrupt(self, interrupts) -> None:
        if not interrupts:
            return
        for itr in interrupts:
            value = getattr(itr, "value", None) or {}
            kind = value.get("kind")
            if kind == "approval":
                await self._emit(AwaitingApprovalEvent())
            elif kind == "clarification":
                qs_raw = value.get("questions") or []
                questions = [
                    q if isinstance(q, ClarifyingQuestion) else ClarifyingQuestion.model_validate(q)
                    for q in qs_raw
                ]
                await self._emit(ClarifyingQuestionsEvent(questions=questions))

    async def _run_research(self) -> None:
        if self.approved_plan is None:
            await self._emit(ErrorEvent(message="research: no approved plan"))
            await self.queue.put(None)
            return
        try:
            graph = build_research_graph()
            cfg: RunnableConfig = {
                "configurable": {"deps": self.deps, "thread_id": self._research_thread}
            }
            await graph.ainvoke(
                {
                    "user_query": self.user_query,
                    "approved_plan": self.approved_plan,
                    "scout_findings": self.scout_findings,
                },
                config=cfg,
            )
            await self._emit(DoneEvent())
            await self.queue.put(None)
        except Exception as e:
            log.exception("research run failed")
            await self._emit(ErrorEvent(message=f"research failed: {e}"))
            await self.queue.put(None)

    async def aclose(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    # -- read-only state for /state endpoint --------------------------------

    def state_snapshot(self) -> dict:
        return {
            "session_id": self.id,
            "mode": self.mode,
            "locked": self.locked,
            "has_approved_plan": self.approved_plan is not None,
            "scout_findings_count": len(self.scout_findings),
        }


class SessionManager:
    """In-memory registry of sessions. Single-process; not durable."""

    def __init__(
        self, llm: LLMClient, settings: Settings, trace_dir: Path | None = None
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.trace_dir = trace_dir
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex[:12]
        if sid in self._sessions:
            return self._sessions[sid]
        sess = Session(
            session_id=sid,
            llm=self.llm,
            settings=self.settings,
            trace_dir=self.trace_dir,
        )
        self._sessions[sid] = sess
        return sess

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def aclose(self) -> None:
        for s in list(self._sessions.values()):
            await s.aclose()
        self._sessions.clear()
