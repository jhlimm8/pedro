"""Runtime configuration. All values overridable via env vars.

Loads `.env` from the repo root on import so `from_env()` sees vars without
the caller having to remember to do it. Mirrors the mdb pattern: a single
`OpenAI` (or `AsyncOpenAI`) client with `api_key` + `base_url`, where
pointing `base_url` at a LiteLLM proxy is a drop-in replacement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Load .env eagerly. Safe to call repeatedly; no-ops if already loaded
# or if the file doesn't exist.
load_dotenv()

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str | None
    """Override the OpenAI API base URL. Set to e.g. `http://localhost:4000`
    to route through a LiteLLM proxy with a LiteLLM virtual key in
    `OPENAI_API_KEY`. Leave unset to hit api.openai.com directly."""

    model: str
    reasoning_effort: ReasoningEffort
    scout_max_subquestions: int
    deep_max_subquestions: int
    request_timeout_s: int
    trace_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            # Don't raise here; tests that don't hit the network shouldn't need a key.
            # Real network calls will fail with a clear OpenAI error if key is missing.
            api_key = ""

        effort = os.getenv("PEDRO_REASONING_EFFORT", "low").lower()
        if effort not in ("minimal", "low", "medium", "high"):
            effort = "low"

        # Accept either OPENAI_BASE_URL (the official OpenAI SDK env var) or
        # PEDRO_OPENAI_BASE_URL (project-local). Empty string is treated as
        # "unset" so an accidentally-blank line in .env doesn't break the
        # default routing to api.openai.com.
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("PEDRO_OPENAI_BASE_URL")
            or None
        )
        if base_url == "":
            base_url = None

        return cls(
            openai_api_key=api_key,
            openai_base_url=base_url,
            model=os.getenv("PEDRO_MODEL", "gpt-5"),
            reasoning_effort=effort,  # type: ignore[arg-type]
            scout_max_subquestions=int(os.getenv("PEDRO_SCOUT_MAX_SUBQUESTIONS", "5")),
            deep_max_subquestions=int(os.getenv("PEDRO_DEEP_MAX_SUBQUESTIONS", "8")),
            request_timeout_s=int(os.getenv("PEDRO_REQUEST_TIMEOUT_S", "180")),
            trace_dir=os.getenv("PEDRO_TRACE_DIR", "traces"),
        )
