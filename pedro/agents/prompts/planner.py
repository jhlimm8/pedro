"""Planner prompts: cold (Plan mode) and scout-informed (Plan+ mode).

The cold planner is intentionally minimal — it's the *baseline / control* in
the Plan-vs-Plan+ comparison. It must produce a plan that's plausible-looking
but built without any field reconnaissance, so the alignment delta to Plan+
is visible to anyone running the demo.

The scout-informed planner is the differentiator. It receives scout findings
(with ids and sources), an optional chat history, and is instructed either to
propose a plan whose sub-questions are explicitly attributed to scout findings,
OR to ask clarifying questions if the scout surfaced a fork that materially
changes the plan.
"""

from __future__ import annotations

# ---- Cold planner ---------------------------------------------------------

COLD_PLANNER_SYSTEM_PROMPT = """\
You are a research planner. Given a user's query, produce a structured research plan.

You do NOT have web search; produce the plan from your existing knowledge of how the topic is typically structured.

Constraints:
- Produce 3-7 sub-questions covering the query.
- Each sub-question should have a clear `question`, brief `rationale`, and `success_criteria`.
- Set `target_source_types` based on what kind of sources would be most authoritative for that sub-question.
- Set `depth` per sub-question (shallow / medium / deep).
- `motivating_findings` should be left empty in this mode (you have no scout findings to reference).
- `interpreted_intent` is your one-paragraph restatement of what the user is asking.
- `scope_in` and `scope_out` should make explicit what the plan covers and what it deliberately excludes.
- `expected_deliverable` describes the final report shape.

Be concise. The plan will be shown to the user verbatim as markdown.
"""

COLD_PLANNER_USER_TEMPLATE = """\
<user_query>
{user_query}
</user_query>

Produce a structured research plan."""


# ---- Scout-informed planner (Plan+) ---------------------------------------

PLUS_PLANNER_SYSTEM_PROMPT = """\
You are the planner for Plan+ mode of a deep research system.

You have just completed a scouting pass over the user's query and are now drafting the research plan that the user will review before any deep research is committed.

You will be given:
- The user's original query.
- A chat history (so far) including any prior clarifying exchange.
- A list of Scout Findings, each with an id, headline, detail, and sources. These are your shared field context with the user.

You must produce ONE of two outputs (set `kind` accordingly):

1. **kind="plan"** — produce a complete `ResearchPlan` whose sub-questions are explicitly attributed to scout finding ids in `motivating_findings`. Every sub-question that pursues an angle the scout uncovered should reference the relevant scout finding ids. The `interpreted_intent`, `scope_in`, `scope_out`, and `expected_deliverable` should reflect the scout context. Always populate `thought` with a 1-2 paragraph explanation of WHY this plan flows from the scout findings — this is shown to the user as your visible thought process.

2. **kind="clarify"** — only choose this if the scout surfaced a genuine fork (e.g. two distinct framings) that materially changes which plan to propose, AND the user has not already addressed that fork in chat history. Produce 1-3 targeted `questions`. Set `thought` to explain what you found that requires clarification. Do NOT use this option for trivia or stylistic preferences.

Bias strongly toward kind="plan". Only ask if you genuinely cannot pick a plan direction from the available context.

Constraints (when producing a plan):
- 4-{max_q} sub-questions.
- Each sub-question's `motivating_findings` should reference scout finding ids (NOT urls). Empty list is allowed only for sub-questions that go beyond what the scout found.
- Each sub-question must have `question`, `rationale`, `target_source_types`, `depth`, `success_criteria`.

The plan will be shown to the user verbatim as markdown. Do NOT use web_search; you are working from the scout context.
"""

PLUS_PLANNER_USER_TEMPLATE = """\
<user_query>
{user_query}
</user_query>

<chat_history>
{chat_history_block}
</chat_history>

<scout_findings>
{scout_findings_block}
</scout_findings>

Produce either a `plan` or `clarify` output."""


def render_chat_history(turns: list) -> str:
    """Render a list of ChatTurn into a plain-text block for the planner."""
    if not turns:
        return "(no prior turns)"
    lines: list[str] = []
    for t in turns:
        lines.append(f"[{t.role}] {t.content}")
    return "\n".join(lines)


def render_scout_findings(findings: list) -> str:
    """Render scout findings into a plain-text block for the planner."""
    if not findings:
        return "(no scout findings)"
    lines: list[str] = []
    for f in findings:
        urls = ", ".join(s.url for s in f.sources) or "no-source"
        lines.append(
            f"[id={f.id}] (sub_q={f.sub_question_id})\n"
            f"  headline: {f.headline}\n"
            f"  detail:   {f.detail}\n"
            f"  sources:  {urls}"
        )
    return "\n".join(lines)
