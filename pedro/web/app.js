// PEDRO web client. Single-file vanilla JS — no build step.
//
// The UI is laid out as a typeset ledger:
//   - Each chat turn is a row with a left "gutter" (time · role) and a right
//     "content" column.
//   - Findings are rendered as quoted dispatches with a source-tier badge.
//   - The mode selector (Plan / Plan+ / Research) is INLINE with the composer
//     and is cycled with Tab / Shift+Tab while the textarea is focused.
//
// Mode-lock semantics:
//   - Before any plan is approved, Plan and Plan+ are freely switchable.
//     Research can be selected but is non-runnable until a plan exists.
//   - Once a plan is approved, the planning modes are locked (struck through)
//     and Research is selected automatically for the rest of the session.

const API = "/api";

const state = {
  sessionId: null,
  mode: "plan",
  locked: false,
  hasPlan: false,
  awaiting: null,           // null | "approval" | "clarification"
  pendingClarifyQs: [],
  eventSource: null,
};

// ---------- DOM ------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);

const chat            = $("#chat");
const composer        = $("#composer");
const messageInput    = $("#message");
const sendBtn         = $("#send");
const actionPanel     = $("#action-panel");
const sessionIdEl     = $("#session-id");
const newSessionBtn   = $("#new-session");
const statusDot       = $("#status-dot");
const statusText      = $("#status-text");
const statusModeEl    = $("#status-mode");

const MODES = ["plan", "plan_plus", "research"];

const MODE_LABELS = {
  plan: "plan",
  plan_plus: "plan+",
  research: "research",
};

const ROLE_TIME_FMT = (d) =>
  `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;

// ---------- helpers --------------------------------------------------------

function escapeHtml(s) {
  return (s || "")
    .toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(s) {
  if (window.marked) return window.marked.parse(s || "");
  return `<p>${escapeHtml(s || "")}</p>`;
}

function setStatus(kind, text) {
  statusDot.className = `dot dot-${kind}`;
  statusText.textContent = text;
}

function setStatusMode() {
  statusModeEl.textContent = MODE_LABELS[state.mode] || state.mode;
}

// ---------- Ledger rows ----------------------------------------------------

function addEntry({ role, content, klass = "", html = false }) {
  const div = document.createElement("article");
  div.className = `ledger-entry ${klass}`.trim();
  const when = ROLE_TIME_FMT(new Date());
  const body = html ? content : escapeHtml(content);
  div.innerHTML = `
    <div class="gutter">
      <span class="time mono">${when}</span>
      <span class="role">${escapeHtml(role)}</span>
    </div>
    <div class="content">${body}</div>
  `;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

const userEntry      = (text)       => addEntry({ role: "you",     content: `<p>${escapeHtml(text)}</p>`, klass: "user", html: true });
const systemEntry    = (html)       => addEntry({ role: "system",  content: html, klass: "system", html: true });
const scoutEntry     = (html)       => addEntry({ role: "scout",   content: html, klass: "scout", html: true });
const planEntry      = (html)       => addEntry({ role: "plan",    content: html, klass: "plan", html: true });
const researchEntry  = (html)       => addEntry({ role: "research",content: html, klass: "research", html: true });
const reportEntry    = (html)       => addEntry({ role: "report",  content: html, klass: "report", html: true });

// ---------- Mode tabs ------------------------------------------------------

const modeTabs = document.querySelectorAll(".mode-tab");

function selectableModes() {
  if (state.locked) return ["research"];
  return ["plan", "plan_plus", "research"];
}

function refreshModeTabs() {
  modeTabs.forEach((tab) => {
    const m = tab.dataset.mode;
    const isSelected = m === state.mode;
    tab.setAttribute("aria-selected", isSelected ? "true" : "false");

    let locked = false;
    if (state.locked) {
      // Plan approved: only research is selectable.
      locked = m !== "research";
    } else {
      // Before approval: research is selectable but unrunnable; we don't grey
      // it out in the tab strip — it's a *visible* future state. The send
      // button is the gate.
      locked = false;
    }
    tab.classList.toggle("locked", locked);
  });

  const composerBlocked =
    state.mode === "research" && !state.hasPlan && !state.locked;
  sendBtn.disabled = composerBlocked || state.awaiting !== null;
  messageInput.disabled = composerBlocked;
  if (composerBlocked) {
    messageInput.placeholder =
      "Approve a plan in Plan or Plan+ first — Research executes an approved plan.";
  } else if (state.locked) {
    messageInput.placeholder = "Send a follow-up to refine the report…";
  } else {
    messageInput.placeholder = "Pose a question to the operator…";
  }

  setStatusMode();
}

function setMode(next) {
  if (!MODES.includes(next)) return;
  if (state.locked && next !== "research") return;
  state.mode = next;
  refreshModeTabs();
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    if (tab.classList.contains("locked")) return;
    setMode(tab.dataset.mode);
    messageInput.focus();
  });
});

// Tab / Shift+Tab cycle modes when the composer textarea has focus.
// We intercept Tab unconditionally here (matching Cursor's chat behaviour) —
// users who actually need to escape the textarea can click outside.
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Tab" && !e.altKey && !e.metaKey && !e.ctrlKey) {
    e.preventDefault();
    const allowed = selectableModes();
    if (allowed.length === 0) return;
    const idx = allowed.indexOf(state.mode);
    const start = idx === -1 ? 0 : idx;
    const dir = e.shiftKey ? -1 : 1;
    const nextIdx = (start + dir + allowed.length) % allowed.length;
    setMode(allowed[nextIdx]);
    return;
  }
  // Cmd/Ctrl+Enter or plain Enter (without shift) submits.
  if (e.key === "Enter" && !e.shiftKey && !e.altKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

// ---------- Action panel ---------------------------------------------------

function clearActionPanel() {
  actionPanel.innerHTML = "";
  actionPanel.classList.add("hidden");
  state.awaiting = null;
  state.pendingClarifyQs = [];
  refreshModeTabs();
}

function showApprovalActions() {
  state.awaiting = "approval";
  actionPanel.innerHTML = `
    <div class="heading">Awaiting approval</div>
    <div class="body">Approve to dispatch the research run, or send edits and the planner will redraft.</div>
    <div class="action-row">
      <textarea id="edit-text" rows="2" placeholder="Optional: edits or direction for a re-plan…"></textarea>
    </div>
    <div class="action-row">
      <button type="button" id="btn-approve" class="act-btn primary">Approve &amp; dispatch</button>
      <button type="button" id="btn-edit" class="act-btn">Send edits &amp; redraft</button>
    </div>
  `;
  actionPanel.classList.remove("hidden");
  $("#btn-approve").addEventListener("click", onApprove);
  $("#btn-edit").addEventListener("click", onSendEdits);
  refreshModeTabs();
}

function showClarifyActions(questions) {
  state.awaiting = "clarification";
  state.pendingClarifyQs = questions;
  const qsHtml = questions
    .map((q, i) => `<li><b>Q${i + 1}.</b> ${escapeHtml(q.question)}</li>`)
    .join("");
  actionPanel.innerHTML = `
    <div class="heading">The planner has questions</div>
    <ol>${qsHtml}</ol>
    <div class="action-row">
      <textarea id="clarify-text" rows="2" placeholder="Reply to the planner…" autofocus></textarea>
      <button type="button" id="btn-clarify-send" class="act-btn primary">Send reply</button>
    </div>
  `;
  actionPanel.classList.remove("hidden");
  $("#btn-clarify-send").addEventListener("click", onSendClarifyReply);
  setTimeout(() => $("#clarify-text")?.focus(), 0);
  refreshModeTabs();
}

async function onApprove() {
  const sid = state.sessionId;
  if (!sid) return;
  await fetch(`${API}/chat/${sid}/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "approve", text: "" }),
  });
  clearActionPanel();
  setStatus("running", "approved · dispatching research");
}

async function onSendEdits() {
  const sid = state.sessionId;
  if (!sid) return;
  const text = ($("#edit-text").value || "").trim();
  if (!text) {
    alert("Please describe what to change.");
    return;
  }
  userEntry(`Edits: ${text}`);
  await fetch(`${API}/chat/${sid}/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "edit", text }),
  });
  clearActionPanel();
  setStatus("running", "redrafting plan");
}

async function onSendClarifyReply() {
  const sid = state.sessionId;
  if (!sid) return;
  const text = ($("#clarify-text").value || "").trim();
  if (!text) {
    alert("Please answer the clarifying question(s).");
    return;
  }
  userEntry(text);
  await fetch(`${API}/chat/${sid}/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "clarify_reply", text }),
  });
  clearActionPanel();
  setStatus("running", "planner re-running");
}

// ---------- Compose (send) -------------------------------------------------

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  if (sendBtn.disabled) return;

  userEntry(text);
  messageInput.value = "";

  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  setStatus("running", "starting");
  const r = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      mode: state.mode,
      message: text,
    }),
  });
  if (!r.ok) {
    const err = await r.text();
    setStatus("error", `error · ${r.status}`);
    systemEntry(`<b>Error ${r.status}.</b> ${escapeHtml(err)}`);
    return;
  }
  const body = await r.json();
  state.sessionId = body.session_id;
  sessionIdEl.textContent = state.sessionId.slice(0, 8);
  openStream(state.sessionId);
});

// ---------- New session ----------------------------------------------------

newSessionBtn.addEventListener("click", () => {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.sessionId = null;
  state.locked = false;
  state.hasPlan = false;
  state.mode = "plan";
  clearActionPanel();
  chat.innerHTML = "";
  sessionIdEl.textContent = "—";
  setStatus("idle", "at rest");
  refreshModeTabs();
  messageInput.focus();
});

// ---------- SSE event handling --------------------------------------------

function openStream(sid) {
  const es = new EventSource(`${API}/chat/${sid}/stream`);
  state.eventSource = es;

  const handlers = {
    mode_set: onModeSet,
    error: onErrorEvent,
    done: onDone,
    assistant_message: onAssistantMessage,
    scout_started: onScoutStarted,
    scout_sub_question: onScoutSubQuestion,
    scout_finding: onScoutFinding,
    scout_complete: onScoutComplete,
    clarifying_questions: onClarifyingQuestions,
    plan_proposed: onPlanProposed,
    awaiting_approval: onAwaitingApproval,
    plan_approved: onPlanApproved,
    mode_locked: onModeLocked,
    research_started: onResearchStarted,
    subquestion_progress: onSubquestionProgress,
    source_found: onSourceFound,
    deep_finding: onDeepFinding,
    synthesis_started: onSynthesisStarted,
    final_report: onFinalReport,
  };

  for (const [type, fn] of Object.entries(handlers)) {
    es.addEventListener(type, (e) => {
      let data = {};
      try { data = JSON.parse(e.data); } catch (_) {}
      fn(data);
    });
  }

  es.onerror = () => {
    setStatus("error", "stream error");
  };
}

// ---------- finding rendering ---------------------------------------------

function tierBadge(tier) {
  if (!tier) return "";
  const safe = String(tier).toLowerCase();
  if (!["official", "reputable", "common"].includes(safe)) return "";
  return `<span class="tier ${safe}">${escapeHtml(safe)}</span>`;
}

function sourceList(sources) {
  return (sources || [])
    .map((s) => {
      const link = `<a href="${encodeURI(s.url)}" target="_blank" rel="noreferrer">${escapeHtml(s.title || s.url)}</a>`;
      return `${tierBadge(s.tier)}${link}`;
    })
    .join(" &nbsp;·&nbsp; ");
}

function findingHtml({ phase, finding }) {
  const f = finding || {};
  const klass = phase === "research" ? "finding research" : "finding scout";
  const label = phase === "research" ? "deep finding" : "scout finding";
  return `
    <div class="${klass}">
      <div class="head">
        <span class="label">${label}</span>
        <span class="id mono">${escapeHtml(f.id || "")}</span>
        <div class="headline">${escapeHtml(f.headline || "")}</div>
      </div>
      <div class="detail">${escapeHtml(f.detail || "")}</div>
      <div class="src">${sourceList(f.sources)}</div>
    </div>
  `;
}

// ---------- handlers -------------------------------------------------------

function onModeSet(d) {
  systemEntry(`mode <b>${escapeHtml(MODE_LABELS[d.mode] || d.mode)}</b> engaged.`);
}

function onAssistantMessage(d) {
  addEntry({ role: "pedro", content: renderMarkdown(d.content || ""), html: true });
}

function onScoutStarted(d) {
  setStatus("running", `scouting · ${d.sub_question_count} sub-question(s)`);
  scoutEntry(`<p>Sending the scout into the field. <b>${d.sub_question_count}</b> sub-question(s) drafted.</p>`);
}

function onScoutSubQuestion(d) {
  systemEntry(`scout · <span class="mono">${escapeHtml(d.sub_question_id)}</span> — ${escapeHtml(d.question)}`);
}

function onScoutFinding(d) {
  scoutEntry(findingHtml({ phase: "scout", finding: d.finding }));
}

function onScoutComplete(d) {
  systemEntry(`scout returned · <b>${d.finding_count}</b> finding(s) on file.`);
}

function onClarifyingQuestions(d) {
  setStatus("paused", "awaiting clarification");
  showClarifyActions(d.questions || []);
}

function onPlanProposed(d) {
  state.hasPlan = true;
  planEntry(renderMarkdown(d.plan_markdown || ""));
  refreshModeTabs();
}

function onAwaitingApproval(_d) {
  setStatus("paused", "awaiting approval");
  showApprovalActions();
}

function onPlanApproved(_d) {
  systemEntry(`plan approved · handing off to deep research.`);
}

function onModeLocked(_d) {
  state.locked = true;
  state.mode = "research";
  refreshModeTabs();
  systemEntry(`session locked to <b>research</b> · planning modes are now read-only.`);
}

function onResearchStarted(d) {
  setStatus("running", `research · ${d.sub_question_count} sub-question(s)`);
  researchEntry(`<p>Dispatching the research agent. <b>${d.sub_question_count}</b> sub-question(s) in flight.</p>`);
}

function onSubquestionProgress(d) {
  systemEntry(`research · <span class="mono">${escapeHtml(d.sub_question_id)}</span> — ${escapeHtml(d.status)}`);
}

function onSourceFound(d) {
  const s = d.source || {};
  systemEntry(`source · ${tierBadge(s.tier)}<a href="${encodeURI(s.url)}" target="_blank" rel="noreferrer">${escapeHtml(s.title || s.url)}</a>`);
}

function onDeepFinding(d) {
  researchEntry(findingHtml({ phase: "research", finding: d.finding }));
}

function onSynthesisStarted(_d) {
  setStatus("running", "synthesising report");
  systemEntry(`research · synthesising final report…`);
}

function onFinalReport(d) {
  reportEntry(renderMarkdown(d.report_markdown || "*(no report)*"));
  if ((d.contradictions || []).length) {
    const lis = d.contradictions.map((c) => `<li>${escapeHtml(c)}</li>`).join("");
    systemEntry(`<b>Contradictions surfaced:</b><ul>${lis}</ul>`);
  }
}

function onErrorEvent(d) {
  setStatus("error", "error");
  systemEntry(`<b>Error.</b> ${escapeHtml(d.message || "(unknown)")}`);
}

function onDone(_d) {
  setStatus("done", "complete");
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

// ---------- Init -----------------------------------------------------------

refreshModeTabs();
setStatus("idle", "at rest");
