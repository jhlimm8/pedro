"""Synthesizer prompt — produces the final cited markdown report."""

from __future__ import annotations

SYNTHESIZER_SYSTEM_PROMPT = """\
You are the synthesizer for a deep research system. The research phase has completed; your job is to produce the final report that the user will read.

You will receive:
- The interpreted intent.
- The approved plan (sub-questions, scope, deliverable).
- All deep-research findings, grouped by sub-question.

Produce a `report_markdown` string that:
1. Opens with a TL;DR (2-4 sentences) directly addressing the user's intent.
2. Has one section per plan sub-question, in plan order. Use the sub-question text as the section heading.
3. Inside each section, synthesize the findings into prose; cite sources inline using `[N]` numbered references that point to a `## Sources` section at the end.
4. Numbers a single global Sources list (deduplicated by URL) at the end as `1. <title> [<tier>] — <url>`. Include the source's tier (official / reputable / common) in brackets when known so the reader can weight evidence accordingly.
5. If `expected_deliverable` specifies a particular shape (table, comparison, etc.), produce that shape inside or in addition to the prose.

Also produce a `contradictions` list: short strings describing any cross-source contradictions you noticed. Empty list if none. Do not invent contradictions where the sources merely emphasize different aspects.

Be faithful to the findings. Do NOT add facts that aren't in the findings. Do NOT use web_search; you are synthesizing what was already gathered.
"""

SYNTHESIZER_USER_TEMPLATE = """\
<interpreted_intent>
{interpreted_intent}
</interpreted_intent>

<expected_deliverable>
{expected_deliverable}
</expected_deliverable>

<plan_sub_questions>
{plan_sub_questions_block}
</plan_sub_questions>

<findings_by_sub_question>
{findings_block}
</findings_by_sub_question>

Produce the structured output."""


def render_plan_sub_questions(sub_questions: list) -> str:
    if not sub_questions:
        return "(no sub-questions)"
    lines: list[str] = []
    for i, sq in enumerate(sub_questions, 1):
        lines.append(f"{i}. [id={sq.id}] {sq.question}")
        if sq.success_criteria:
            lines.append(f"   success: {sq.success_criteria}")
    return "\n".join(lines)


def render_findings_grouped(deep_findings: list) -> str:
    if not deep_findings:
        return "(no findings)"
    by_sq: dict[str, list] = {}
    for f in deep_findings:
        by_sq.setdefault(f.sub_question_id or "_unattributed", []).append(f)

    lines: list[str] = []
    for sq_id, findings in by_sq.items():
        lines.append(f"--- Sub-question id={sq_id} ---")
        for f in findings:
            urls = "; ".join(
                f"{s.title or '(no title)'} ({s.tier or 'untiered'}) -- {s.url}"
                for s in f.sources
            )
            lines.append(f"  [{f.id}] {f.headline}")
            lines.append(f"      {f.detail}")
            lines.append(f"      sources: {urls or '(no sources)'}")
    return "\n".join(lines)
