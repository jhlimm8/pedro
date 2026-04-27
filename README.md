# PEDRO — Plan-Extended Deep Research Operator

> A multi-agent deep research system whose differentiator is **alignment-first**  
> **planning**: it opts to spend more tokens on building the plan, *before* it spends tokens on deep research.

## TL;DR

Most deep-research agents fail not because they search badly, but because the
plan they execute isn't the plan you wanted. PEDRO treats planning itself as
a mini research task — a "scout" pass that pulls back 2-4 broad findings,
shows them to you, and lets the planner ask clarifying questions before
proposing the actual plan. Approve the plan, and only then does the heavier
deep-research dispatcher fan out.

The headline UX is three tabs:

- **Plan** — control / baseline. Cold planner drafts a plan from your query
alone, the way most agents do.
- **Plan+** — the alignment-first mode. Scout researcher runs first, then a
scout-informed planner proposes a plan that cites the scout's findings.
Can ask clarifying questions mid-loop.
- **Research** — execution. Greyed out until a plan is approved; then
auto-engaged with Plan / Plan+ locked.

The hypothesis is that Plan+ will result in more aligned plans than Plan, which will in turn result in more aligned research.

## Video Demo

[https://github.com/user-attachments/assets/da47d625-dbcd-4178-adfe-14bf8c705eab](https://github.com/user-attachments/assets/da47d625-dbcd-4178-adfe-14bf8c705eab)

## Run it

```bash
# 1) Install runtime deps (uv reads .python-version and creates .venv for you)
uv sync

# 2) Configure
cp .env.example .env
# edit .env to set OPENAI_API_KEY

# 3) Boot the API + UI
uv run pedro --reload
#   defaults: host=127.0.0.1, port=8000 — falls back to any free port if 8000 is taken
#   override: uv run pedro --host 0.0.0.0 --port 8123
#   env vars: PEDRO_HOST, PEDRO_PORT

# UI:    http://127.0.0.1:8000/   (or whatever port it picked — printed at startup)
# API:   http://127.0.0.1:8000/docs
```

The UI is a single-file vanilla HTML+JS chat. No build step.

> Add `--extra dev` to `uv sync` only when you want to run the test suite
> (pytest, ruff, etc). It's not needed to run the app itself.

### CLI / curl

```bash
# Start a Plan+ session
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"mode":"plan_plus","message":"Compare Postgres logical vs physical replication."}'
# -> { "session_id": "...", "mode": "plan_plus", "locked": false }

# Stream events
curl -N localhost:8000/api/chat/<sid>/stream

# Approve the proposed plan -> automatically transitions into Research mode
curl -s localhost:8000/api/chat/<sid>/respond -H 'content-type: application/json' \
  -d '{"action":"approve","text":""}'
```

## Architecture

```
                       +--------------------+
   user query  ───────▶│ POST /api/chat     │
                       +--------------------+
                                  │
                                  ▼
                       +--------------------+
                       │  Session (per-id)  │
                       │  - InMemorySaver   │
                       │  - SSE event queue │
                       │  - mode-lock state │
                       +--------------------+
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
         build_plan_graph   build_plus_graph    build_research_graph
            (Plan)              (Plan+)              (Research)

  Plan:           planner_cold ─▶ approval_gate ─▶ END
                                       │
                                       ▼ (interrupt)
                                  user approves
                                       │
                                       ▼
                                Research graph

  Plan+:  scout_node ─▶ planner_plus ─┬─▶ clarify_gate ─▶ planner_plus ─▶ ...
                                      └─▶ approval_gate ─▶ END
                                                │
                                                ▼ (interrupt)
                                           user approves
                                                │
                                                ▼
                                          Research graph

  Research:  dispatcher (fan-out research_one×N) ─▶ synthesizer ─▶ END
```

The **researcher primitive** (`pedro/agents/nodes/researcher.py`) is one
function shared by both phases:

- `phase="scout"` — broad/shallow, 2-4 findings per call, used by `scout_node`
- `phase="deep"`  — focused/thorough, 3-6 findings per call, used by `dispatcher_node`

Same prompt structure, same tool wiring (OpenAI Responses API +
`web_search_preview`), different system prompts and stricter source-tier
guidance for deep research.

### Mode-lock semantics (implementation reference)

A session has a single bit, `locked`:

- `**locked=False`** (no plan approved): user can switch between Plan and
Plan+ tabs; Research is selectable but the composer is disabled.
- `**locked=True`** (plan approved): Plan / Plan+ tabs are visually locked;
research has auto-engaged and is running. The UI gets a `mode_locked`
SSE event and switches to the Research tab.

The rule is enforced in three places (defense in depth):

1. `**Session.start()`** — refuses to start a planning mode if `locked`,
  refuses to start `research` mode if not `locked`.
2. `**refreshModeTabs()` in the UI** — visually locks tabs and disables
  the composer.
3. **The graphs themselves** — Plan / Plan+ write `approved_plan` into
  shared state; the Research graph reads it. There is no path that runs
   Research without an approved plan in state.

## Design decisions

This section documents every non-trivial architectural choice in PEDRO.
Each entry follows a tight ADR shape: **context → decision → alternatives
→ why → tradeoffs**. If you only have time for the headline ones, read
D1, D2, D3, D5, D6, and D8.

### D1. Frame the problem as alignment, not retrieval quality

**Context.** The brief is "build a multi-agent deep-research system." The
default failure mode for systems of this kind in the wild is *misalignment*,
not bad search: the agent runs a competent but off-target plan, returns a
polished report on the wrong question, and the user has to start over.

**Decision.** Position PEDRO as an alignment-first system. The differentiator
is the planning UX: scout-first planning + an explicit reviewable plan
artifact + sign-off-or-edit *before* any deep research runs.

**Alternatives considered.** Better retrieval (rerankers, multi-source
search), more parallelism (bigger fan-out), agent specialization (separate
domain agents). All of these address *quality of execution*; none address
misalignment.

**Why this.** The brief's correctness criterion is *output relative to user
intent*, and the most expensive failure mode is a polished report on the
wrong question. A 30 sec planning re-roll dominates a 5 min wrong-direction
research run. The headline test (Plan vs Plan+ on the same query) is
designed to make this concrete.

**Tradeoff.** PEDRO sometimes spends a beat on the plan when the user knows
exactly what they want — Plan mode (cold planner) is the explicit escape
hatch for that case.

### D2. LangGraph with a checkpointer + interrupts (stateful graph)

**Context.** Plan-review HITL means the agent has to *pause* mid-execution
waiting for user input, then *resume* with state intact. Often more than
once (clarifying questions → answer → plan → edit → re-plan → approve).

**Decision.** Use LangGraph's `StateGraph` with an `InMemorySaver`
checkpointer and `interrupt()` at every HITL boundary. Resume via
`Command(resume=...)`.

**Alternatives considered.**

- Stateless prompt chain (re-run from scratch on each user turn). Forces
the planner to re-derive context every turn, multiplies tokens, and can't
represent "the planner is paused waiting for an answer to *this specific
question*."
- Hand-rolled state machine with a class hierarchy. Doable, but reinvents
`interrupt`, edge routing, and replay-from-checkpoint.
- LangChain agents / CrewAI / AutoGen. These are loop-driven (ReAct-ish);
they don't have first-class interrupt-and-resume primitives, and they
hide graph topology behind agent personas.

**Why this.** `interrupt()` is the exact primitive plan-review HITL needs.
The checkpointer makes re-entry on `respond()` correct by construction —
we don't manually persist + reconstitute partial state. Mode handoff is
just composition: Plan / Plan+ write `approved_plan` into shared state;
the Research graph reads it.

**Tradeoff.** LangGraph adds an opinionated dependency that's still young
(1.x). We lean on a few advanced features (`interrupt`, `Command(resume)`,
configurable Deps). Worth it for the alignment we get on the HITL primitive.

### D3. Server-Sent Events for narration

**Context.** The user needs to *watch* the planning and research happen —
sub-questions being drafted, sources being found, findings being filed,
the synthesizer composing the report. That narration is one-way (server →
client) and arrives in bursts over a long-lived connection.

**Decision.** SSE over plain HTTP via `sse-starlette`, one EventSource per
session. The client drives mutations through normal POSTs (`/chat`,
`/respond`); the server narrates over the SSE stream.

**Alternatives considered.**

- WebSockets. Bidirectional, but we don't *need* bidirectional. Adds an
axis of state (an open socket) we'd otherwise not have, and traverses
fewer reverse proxies cleanly.
- HTTP long-polling. Higher latency, more complex client logic, no native
reconnect.
- gRPC streaming. Browsers don't speak it natively; would force a JS
proxy.
- Pure REST + UI polls `/state`. Loses the narrative quality entirely;
the UX point is *watching* the agent work.

**Why this.** Narration is one-way. SSE is the simplest thing that fits:
HTTP semantics (proxies, caching, auth all work normally), browser's
`EventSource` does reconnection for free, typed events are just labelled
strings. `sse-starlette` integrates cleanly with FastAPI; we set
`ping=86_400` so heartbeats don't spam the client.

**Tradeoff.** Some corporate proxies buffer SSE responses; we set
`Cache-Control: no-cache` on the static assets and don't enable any
nginx-style buffering. SSE is constrained by the per-origin HTTP/1.1
connection limit (6); not a problem at our scale.

### D4. OpenAI Responses API + built-in `web_search_preview`

**Context.** The agent needs LLM calls *and* a search tool. Two paths: use
the LLM provider's built-in tools, or stitch in a separate search API.

**Decision.** `client.responses.parse(...)` with
`tools=[{"type": "web_search_preview"}]` and a Pydantic
`text_format=ResearcherOutput` (or similar) for structured output. One
provider, one key.

**Alternatives considered.**

- Chat Completions API + Tavily / Serper / Exa. Two providers, two API
surfaces, two keys, two failure modes, two billing accounts. We'd also
format + ground search results ourselves.
- Anthropic / Gemini with their respective web search. Equivalent shape;
we picked OpenAI because the assessment reference (`mdb`) uses it and
we're matching that pattern.
- Self-hosted retrieval (vector DB + crawler). Way out of scope; the
search tool *is* the retrieval system here.

**Why this.** One provider keeps PEDRO operationally self-contained.
Structured output (`text_format=PydanticSchema`) gives us validated
parses for free — the LLM is bound to return shapes we can dispatch on.
Mirrors a known-good pattern from `mdb`.

**Tradeoff.** We give up control over result ranking, source filtering,
and recency. For a V0 demo of the *planning UX* this is the right tradeoff;
if PEDRO ever competes on pure search quality, we'd swap in a dedicated
search layer behind the same `LLMClient` Protocol.

### D5. Three explicit modes (Plan / Plan+ / Research) with mode-lock

**Context.** Most research agents have an implicit mode: you submit a
query, it does *something* — sometimes a one-shot answer, sometimes
plan-then-execute, sometimes hidden behind a settings panel. Users can't
compare strategies on the same query, and they can't tell from the UI
whether the plan has been approved or whether they're past the point of
no return.

**Decision.** Make the mode an explicit, visible affordance. Three tabs
inline with the composer: Plan / Plan+ / Research. Once a plan is approved,
the planning tabs lock and Research auto-engages for the rest of the
session. Tab / Shift+Tab cycle modes from the textarea.

**Alternatives considered.**

- A single auto-mode that picks Plan vs Plan+ vs nothing based on query
complexity. Hides the very thing we want to surface.
- A settings dropdown. Discoverable once, forgotten the second time.
- An inline command palette (`/plan-plus`). Power-users only.

**Why this.** Mode-lock turns the workflow into a contract: you cannot
*accidentally* run research without approving a plan, and you cannot
*accidentally* re-plan after research has started. The locked tabs are
visible read-only history of the contract.

**Tradeoff.** Three tabs add UI surface. We compensate by putting them
in-the-composer (you don't have to look up to find them) and Tab-cyclable
from the textarea (zero-mouse switching, like Cursor's chat).

### D6. Scout phase before the Plan+ planner

**Context.** A planner with no domain knowledge produces generic plans.
Generic plans are exactly the plans users skip-or-skim because there's
no concrete signal to react to.

**Decision.** Plan+ runs `scout_node` first: 2-4 broad/shallow sub-questions,
fanned out via the researcher primitive, returning attributed findings.
The planner then drafts with those findings in context and may ask
clarifying questions before finalising.

**Alternatives considered.**

- Cold planner with self-reflection (drafts → critiques → redrafts).
Reflection without grounding tends to converge on whatever the planner
already believed.
- Multi-pass planner with no scout (just longer thinking). Same problem.
- Always run the full deep research and let the synthesizer do the
alignment work. That's exactly the thing we're trying to avoid;
alignment after the fact is too late.

**Why this.** Mirrors how a human researcher actually works: skim the
field, ask sharp questions, *then* propose a plan. Scout findings give
the user something concrete to react to ("here's what's out there; given
this, the plan is X"), which is the whole point of the alignment-first
framing in D1.

**Tradeoff.** Plan+ is slower and more expensive than Plan by ~1 LLM
round + ~1-3 search-grounded calls. That's the tradeoff Plan mode (the
cold-planner control) exists to make explicit.

### D7. One `research_one` primitive, parameterised by phase

**Context.** Scout calls and deep-research calls do the same thing at
different breadths/depths.

**Decision.** A single function `research_one(sub_question, phase, deps)`
with `phase ∈ {"scout", "deep"}`, two prompt variants, the same tool
wiring, the same Pydantic output schema (`ResearcherOutput`).

**Alternatives considered.**

- Separate `scout_research_one` and `deep_research_one` implementations.
Tempting, but they'd drift — one would get a fix, the other wouldn't.
- A `Researcher` class hierarchy with subclassing. More ceremony, no
extra clarity.

**Why this.** Forces the conceptual claim ("scout vs deep is a parameter,
not a different agent") into the type system. If we ever need a third
phase (e.g. "verify"), it's another parameter, not another implementation.

**Tradeoff.** One function with branching prompts is slightly less
browseable than two named functions. Worth it for the invariants we get.

### D8. No critic, no ambiguity-assessor (the negative-space decision)

**Context.** Multi-agent design literature loves piling on roles: planner,
critic, reviewer, ambiguity-assessor, refiner, etc. Each role costs at
least one LLM round, and each role has to be inductively justified.

**Decision.** PEDRO has no critic and no ambiguity-assessor. The planner
can ask clarifying questions natively (its output is a discriminated
union: `kind="plan"` or `kind="clarify"`); the scout findings act as
grounding so the planner doesn't *need* a critic.

**Alternatives considered.** Both roles were explicitly considered and
removed. The inductive-bias test was: "does this role solve a problem
that follows from our problem statement?" For a critic, no — it would
solve "planner produces bad plans," which isn't our claim. For an
assessor, no — the planner can already pause and ask.

**Why this.** Inductive bias is the engineering discipline of not adding
components without justification. If a failure mode emerges that a critic
would solve, we'd add one — but tied to evidence, not to the shape of
some textbook multi-agent diagram.

**Tradeoff.** PEDRO can produce plans the user disagrees with. The
recovery path is `edit <text>` (loops back to the planner) or restart.
We accept this; the demo shows the Plan+ plan is usually well-aligned
on first try.

### D9. Pydantic v2 schemas as the universal contract

**Context.** Five surfaces have to agree on shape: LLM I/O, graph state,
SSE wire events, REST request/response, frontend rendering.

**Decision.** Pydantic v2 models are the single source of truth for all
five. The same `Source` model the LLM is constrained to produce is what
gets serialised into the SSE event the frontend consumes.

**Alternatives considered.** Plain dicts with manual JSON Schema. Possible,
but every layer has to re-validate; the LLM can't be type-bound to a dict,
only to a class.

**Why this.** With `client.responses.parse(text_format=ResearcherOutput)`,
the LLM is *forced* to return a parseable shape; if it can't, we get a
typed exception. Discriminated unions
(`SSEEvent = Annotated[Union[...], Field(discriminator="event")]`) give us
pattern-matchable wire events with no manual switch statements.

**Tradeoff.** We're tied to Pydantic v2. Worth it.

### D10. Markdown plan as the artifact (not a JSON tree)

**Context.** The plan has to be reviewable and editable. The user has to
be able to skim it, point at a section, ask for changes.

**Decision.** The plan is a markdown document with a fixed structure
(*Interpreted intent / Scope / Sub-questions / Expected deliverable*).
Stored as a plain string in `ResearchPlan.markdown`; rendered in the UI
via `marked.js`.

**Alternatives considered.**

- A JSON tree of plan nodes. Type-safe, but requires custom UI to render
and a custom editor to change.
- A free-form plan blob. No structural guarantees the synthesizer can
rely on.

**Why this.** Markdown is what a human reads. The structure inside it is
enforced by the planner prompt + a regex-light parse for the sub-question
list (which the dispatcher reads). Edits are just text — `edit <text>`
re-prompts the planner with the previous draft and the user's diff.

**Tradeoff.** Sub-question parsing is heuristic. If the planner produces
a markdown variant we don't expect, we degrade to "no sub-questions
extracted, run synthesizer on the full text." Rare in practice and
recoverable.

### D11. `Deps` DI container (no module-level imports for LLM/settings)

**Context.** Tests need to swap the LLM client for a `FakeLLMClient` with
deterministic, queue-based responses. They also need to swap the SSE
emitter for a list-appender to assert event ordering.

**Decision.** Each node accepts a single `deps: Deps` carrying
`llm: LLMClient`, `settings: Settings`, and `emit: Callable[[SSEEvent], None]`.
Threaded through via LangGraph's `RunnableConfig.configurable`.

**Alternatives considered.**

- Module-level imports for the OpenAI client + settings. Forces tests to
monkey-patch globals — brittle.
- A bigger DI container (FastAPI's `Depends`, lagom, etc.). Overkill at
this size.

**Why this.** Tests get a clean seam, prod code gets two named arguments.
Same pattern lets us route through LiteLLM in prod with no code change
(just `OPENAI_BASE_URL=http://localhost:4000`).

### D12. Vanilla HTML/JS for the V0 (no React)

**Context.** The UI surface is small: chat list, composer with mode tabs,
action panel. No app shell, no routing, no state library worth its weight.

**Decision.** Three files in `pedro/web/` (`index.html`, `app.js`,
`style.css`). No build step. `marked.js` from a CDN for markdown
rendering.

**Alternatives considered.** Next.js / Vite + React / SvelteKit. Each
adds a build pipeline, a `node_modules` tree, and a deploy story for the
frontend separately from the backend.

**Why this.** Anyone reviewing the code reads three files. The dev loop
is "save and refresh." Migrating to React later, if surface complexity
grows, is a 1-day project — none of the API contract changes.

**Tradeoff.** State management is `const state = { ... }` and manual DOM
updates. Fine at this scale; would not survive 10 more features.

### D13. Mode tabs inline with the composer (Tab/Shift+Tab cyclable)

**Context.** Most chat UIs put mode/model selectors in a top nav. The
selector is far from the textarea, the keyboard switch is buried in a
command palette, and lock state isn't visible at the point of action.

**Decision.** The three mode tabs sit *above the textarea inside the
composer card*. Active mode shows a 2px cinnabar underline; locked modes
are struck through. While the textarea is focused, plain `Tab` cycles
forward, `Shift+Tab` cycles backward; locked modes are skipped.

**Alternatives considered.** Top-nav tabs (the original V0). `Cmd+.`
command palette (Cursor-like). A dropdown next to Send.

**Why this.** Putting mode-state at the point-of-action means the user
sees lock state at the moment they're about to commit. Tab-cycling matches
Cursor's chat affordance, which the target reviewer is already familiar
with.

**Tradeoff.** We capture `Tab` from the textarea, which preempts the
default tab-out behaviour. Shift+Tab still works, and clicking outside
de-focuses normally. Worth it for the keyboard switch.

### D14. In-memory `SessionManager` (no Redis, no DB)

**Context.** A single dev/demo process serving one user at a time.

**Decision.** `SessionManager` holds a `dict[str, Session]` in-memory;
LangGraph uses `InMemorySaver` for checkpoints; SSE events are an
in-process `asyncio.Queue`.

**Alternatives considered.** Redis-backed checkpointer + session store.
Production-ready, irrelevant at our scale.

**Why this.** Zero infra. The Protocol seam is the same — LangGraph ships
a Redis checkpointer; the day we need horizontal scale, we swap the saver
and add a Redis URL. No node code changes.

**Tradeoff.** Session state is lost on process restart. Acceptable for a
demo; the JSONL trace files give us replay fidelity for post-mortems.

### D15. Source-tier tagging at the prompt (not post-hoc heuristics)

**Context.** Not all sources deserve equal weight. Official docs >
established journalism > random blogs > AI-generated content. The
synthesizer needs *some* signal to bias citation weight.

**Decision.** Both researcher prompts ask the LLM to tag each source
with `tier ∈ {"official", "reputable", "common"}` *at the moment it's
recording the finding*. Synthesizer surfaces the tier in the final Sources
list as `1. Title [official] — url`.

**Alternatives considered.**

- Post-hoc URL heuristics (allowlist of `.gov` / `nature.com` / etc.).
Brittle, doesn't generalise.
- A separate "evaluator" LLM pass that re-tags every source. One more
round-trip, redundant.
- Don't tag at all. The reader has to guess.

**Why this.** The model already has the source in context when it's
drafting the finding; it's the cheapest possible moment to ask "what kind
of source is this?" Mirrors `mdb`'s pattern.

**Tradeoff.** Tier tags are LLM-judged, so they're stochastic. We accept
"wrong tier sometimes" over "no tier ever."

### Smaller decisions (briefly)

- `**uv` over pip-tools/poetry/conda.** One tool reads `.python-version`
for Python pinning, `pyproject.toml` for deps, and writes `uv.lock` for
reproducibility. Industry default in AI Python tooling now.
- **Dynamic port fallback** in `pedro --reload`. Tries 8000, falls back
to any free port. Avoids the "address already in use" dance after a
stale uvicorn.
- `**OPENAI_BASE_URL` indirection.** Lets us route through LiteLLM with
the same code path; mirrors `mdb`.
- **JSONL session traces.** One file per session, each line a typed SSE
event. Replayable, diffable, grep-able.
- `**color-scheme: light only` + `Cache-Control: no-cache`.** UI defenses
against (a) Chrome's auto-dark flag inverting the parchment theme and
(b) a stale browser cache masking new code during dev.

## Add-ons

These are off-core but useful:

- **Source-tier tagging.** Both researcher prompts ask the LLM to tag each
source as `official` / `reputable` / `common`. The synthesizer surfaces
the tier in the final Sources list so the reader can weight evidence.
- **Contradiction surfacing.** `SynthesizerOutput.contradictions` is a
list the synthesizer fills with cross-source disagreements it noticed.
Surfaced in the UI as a separate panel.
- **JSONL session trace.** If `PEDRO_TRACE_DIR` is set, every SSE event
for a session is appended as JSONL to `<dir>/<session_id>.jsonl`.
Useful for replays, analysis, and post-mortem reviews.

## Tests

```bash
# Install dev deps (pytest, ruff, etc) once
uv sync --extra dev

# Unit + fast integration (deterministic; uses FakeLLMClient)
uv run pytest

# Real-API smoke test (gated)
export OPENAI_API_KEY=sk-...
export PEDRO_RUN_SMOKE=1
uv run pytest tests/smoke -s
```

The test layout:

- `tests/unit/` — schema round-trips, FakeLLMClient, individual nodes
(researcher, scout, both planners, dispatcher, synthesizer), SSE
serialization.
- `tests/integration/test_graphs.py` — full state-machine tests of each
sub-graph: Plan + edit-loop + approve, Plan+ + clarify-then-approve,
Plan+ → Research handoff.
- `tests/integration/test_api_sse.py` — Session-level SSE event ordering
for the headline Plan+ → Research pipeline, plus route-level smoke.
- `tests/smoke/test_real_api_smoke.py` — env-gated end-to-end against the
real OpenAI API.

The **dependency-injection seam** is `pedro.agents.deps.Deps`, which carries
the LLM client, settings, and an `emit` callback. Production code passes a
real `OpenAILLMClient` and the real session emitter; tests pass a
`FakeLLMClient` and a list-appending emit. No node directly imports the
session or the OpenAI client.

## Project layout

```
pedro/
  agents/
    nodes/                # graph nodes (one per file)
    prompts/              # prompts (one per agent role)
    schemas.py            # all Pydantic models (state + LLM I/O)
    state.py              # GraphState TypedDict + reducers
    graph.py              # build_plan_graph / build_plus_graph / build_research_graph
    deps.py               # Deps DI container
  api/
    app.py                # FastAPI factory
    routes.py             # /chat, /stream, /respond, /state
    session.py            # Session, SessionManager, _run_planning, _run_research
    sse.py                # typed SSE event protocol + serializer
  llm/
    client.py             # LLMClient Protocol, OpenAILLMClient, FakeLLMClient
  web/
    index.html            # single-file vanilla HTML+JS UI
    app.js
    style.css
  config.py
tests/
pyproject.toml
.env.example
```

## License

MIT.