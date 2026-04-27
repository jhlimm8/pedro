"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope


class _NoCacheStaticFiles(StaticFiles):
    """Static-file mount that asks browsers to revalidate every request.

    During dev (and a single-user local app, which is the V0 deployment shape)
    we never want a stale cached UI bundle masking new code. We send
    `Cache-Control: no-cache` so the browser keeps the file but always
    revalidates with `If-Modified-Since`/`If-None-Match`.
    """

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

from pedro.api.routes import get_router
from pedro.api.session import SessionManager
from pedro.config import Settings
from pedro.llm.client import LLMClient, OpenAILLMClient


def create_app(
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Create the FastAPI app. Pass `llm=FakeLLMClient(...)` for tests.

    The static UI at `pedro/web/` is mounted at `/`. Production-grade auth,
    rate limiting, and CORS are out of scope for V0.
    """
    settings = settings or Settings.from_env()
    llm = llm or OpenAILLMClient(settings)

    manager = SessionManager(
        llm=llm,
        settings=settings,
        trace_dir=Path(settings.trace_dir) if settings.trace_dir else None,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await manager.aclose()

    app = FastAPI(
        title="PEDRO — Plan-Extended Deep Research Operator",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.session_manager = manager

    app.include_router(get_router(), prefix="/api")

    # Static frontend (vanilla HTML+JS).
    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", _NoCacheStaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> FileResponse:
            return FileResponse(
                str(web_dir / "index.html"),
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    return app


def _find_free_port(host: str = "127.0.0.1", preferred: int | None = None) -> int:
    """Return a free TCP port on `host`.

    If `preferred` is set, try it first; if it's already in use, fall back to
    asking the kernel for any free port (port 0 = "pick any"). This avoids
    the classic dev-loop annoyance where a stale uvicorn keeps you locked
    out of 8000.
    """
    import socket

    def _is_free(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
            except OSError:
                return False
            return True

    if preferred is not None and _is_free(preferred):
        return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main() -> None:  # pragma: no cover
    """Boot uvicorn against `create_app`.

    Port resolution:
      1. `--port N` on the CLI wins (uvicorn's own arg parsing).
      2. `PEDRO_PORT` env var.
      3. 8000 if free, otherwise an OS-assigned free port.

    Host resolution: `--host` / `PEDRO_HOST` / `127.0.0.1`.
    """
    import argparse

    import uvicorn

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(prog="pedro")
    parser.add_argument(
        "--host",
        default=os.getenv("PEDRO_HOST", "127.0.0.1"),
        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind. Defaults to PEDRO_PORT or 8000; if taken, picks a free port.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn's reloader (dev only).",
    )
    args = parser.parse_args()

    requested = args.port if args.port is not None else int(os.getenv("PEDRO_PORT", "8000"))
    port = _find_free_port(args.host, preferred=requested)
    if port != requested:
        logging.warning(
            "port %d is in use — falling back to free port %d", requested, port
        )

    print(f"  pedro: serving on http://{args.host}:{port}")
    uvicorn.run(
        "pedro.api.app:create_app",
        factory=True,
        host=args.host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
