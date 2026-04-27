"""FastAPI routes for PEDRO.

Endpoints:
  POST /chat                 -> start or continue a session
  GET  /chat/{sid}/stream    -> SSE stream of events for the session
  POST /chat/{sid}/respond   -> resume the paused planning graph
  GET  /chat/{sid}/state     -> mode-lock state snapshot
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from pedro.agents.schemas import Mode
from pedro.api.session import Session, SessionManager
from pedro.api.sse import serialize_sse


class StartChatRequest(BaseModel):
    session_id: str | None = None
    mode: Mode
    message: str = Field(..., description="User query for plan/plan_plus, or empty for research")


class StartChatResponse(BaseModel):
    session_id: str
    mode: Mode
    locked: bool


class RespondRequest(BaseModel):
    action: Literal["approve", "edit", "clarify_reply"]
    text: str = ""


def get_router(manager_attr: str = "session_manager") -> APIRouter:
    """Build the API router. The FastAPI app must set
    `app.state.<manager_attr> = SessionManager(...)` before serving."""
    router = APIRouter()

    def _manager(request: Request) -> SessionManager:
        m = getattr(request.app.state, manager_attr, None)
        if m is None:
            raise HTTPException(500, f"SessionManager not configured at app.state.{manager_attr}")
        return m

    def _require_session(request: Request, sid: str) -> Session:
        sess = _manager(request).get(sid)
        if sess is None:
            raise HTTPException(404, f"session {sid} not found")
        return sess

    @router.post("/chat", response_model=StartChatResponse)
    async def start_chat(req: StartChatRequest, request: Request) -> StartChatResponse:
        manager = _manager(request)
        sess = manager.get(req.session_id) if req.session_id else None
        if sess is None:
            sess = manager.create(req.session_id)
        try:
            await sess.start(req.mode, req.message)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return StartChatResponse(session_id=sess.id, mode=req.mode, locked=sess.locked)

    @router.get("/chat/{sid}/stream")
    async def stream_chat(sid: str, request: Request):
        sess = _require_session(request, sid)

        async def event_iter():
            async for event in sess.stream_events():
                yield serialize_sse(event)

        # ping=None disables sse-starlette's keep-alive comments; the test
        # parser already skips comments, but turning them off makes streams
        # closer to deterministic in tests and reduces wire noise in dev.
        return EventSourceResponse(event_iter(), ping=86_400)

    @router.post("/chat/{sid}/respond")
    async def respond(sid: str, req: RespondRequest, request: Request) -> dict:
        sess = _require_session(request, sid)
        if req.action == "approve":
            payload: dict = {"action": "approve"}
        elif req.action == "edit":
            payload = {"action": "edit", "edits": req.text}
        else:  # clarify_reply
            payload = {"text": req.text}
        try:
            await sess.respond(payload)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True, "session_id": sid}

    @router.get("/chat/{sid}/state")
    async def get_state(sid: str, request: Request) -> dict:
        sess = _require_session(request, sid)
        return sess.state_snapshot()

    return router
