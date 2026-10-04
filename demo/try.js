/* "Try it": run the real CGLC controller on your own documents.
   All dynamic text is inserted with textContent (never innerHTML). */
(function () {
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const API = (window.CGLC_API_BASE || "").replace(/\/$/, "");
const MAX_DOCS = 20, MAX_CHARS = 600000;
const state = { docs: [], lastResult: null, pdfReady: null };
// Same text as service.DEFAULT_OBLIGATION. Pre-filled so the contract is visible and editable.
const DEFAULT_REQ = "Every claim in the answer is supported by a passage from the supplied documents; no claim is left unsupported.";

function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}

/* ---------- samples ---------- */
const SAMPLES = {
  policy: {
    goal: "How many days per week can employees work remotely, who has to approve it, and how much paid parental leave do they get?",
    docs: [
      { name: "remote-work-policy.txt", text:
        "Remote Work Policy\n\nEmployees may work remotely up to three days per week. The remaining days are spent in the office. " +
        "Remote work requires prior approval from the employee's direct manager; the approval is renewed every six months.\n\n" +
        "Equipment: the company provides a laptop and a monitor. Home-office furniture is not reimbursed. " +
        "Expense claims for internet costs must be filed within 30 days.\n\n" +
        "Security: all laptops must use full-disk encryption, and the VPN is mandatory on public networks." },
      { name: "leave-and-benefits.md", text:
        "# Leave and Benefits\n\nAnnual leave is 25 working days per year. Up to five unused days carry over into the next year.\n\n" +
        "Parental leave: primary caregivers receive 16 weeks of fully paid leave; secondary caregivers receive 4 weeks. " +
        "Leave may be taken in one block or split into two blocks within the first year.\n\n" +
        "Sick leave is unlimited with a doctor's note after three consecutive days." }
    ]
  },
  arch: {
    goal: "Which of the three designs is strongest, and what is the main weakness of each?",
    docs: [
      { name: "design-A.txt", text: "Architecture A uses an always-on semantic coverage layer with selective rich inspection. It supports DEEPEN, VERIFY and REDIRECT interventions. Its weakness is that maintaining coverage state continuously is expensive." },
      { name: "design-B.txt", text: "Architecture B maintains a MAP-like research graph with requirements, evidence links, coverage and marginal gain. Direction awareness is strong. Its weakness is that the controller becomes heavy: it effectively turns into a second research agent." },
      { name: "design-C.txt", text: "Architecture C logs the raw trace and runs semantic checks only at finalization, fixed intervals and stalls. Hard duties sit outside effort judgment, which keeps overhead low. Its weakness is that checkpoint timing is fixed rather than event-driven." }
    ]
  }
};

/* ---------- provider UI ---------- */
const PROV = {
  groq: { model: "openai/gpt-oss-120b", needsKey: true,
          help: ["Get a free key at ", ["console.groq.com/keys", "https://console.groq.com/keys"], ". Any Groq chat model works: load the list, or type a model id."] },
  anthropic: { model: "claude-opus-5-5", needsKey: true,
          help: ["Get a key at ", ["console.anthropic.com", "https://console.anthropic.com/settings/keys"], ". Runs are billed to your account."] },
  offline: { model: "", needsKey: false, help: ["No key, no AI: keyword retrieval only, so you can watch the controller's mechanics on your own text."] }
};
function provider() { return ($("input[name=t-prov]:checked") || {}).value || "groq"; }
function applyProvider() {
  const p = provider(), cfg = PROV[p];
  $("#t-keywrap").hidden = !cfg.needsKey;
  $("#t-modelwrap").hidden = !cfg.needsKey;
  $("#t-load-models").hidden = p !== "groq";
  const m = $("#t-model");
  if (!m.dataset.touched || !m.value) m.value = cfg.model;
  const h = $("#t-keyhelp"); h.textContent = "";
  for (const part of cfg.help) {
    if (Array.isArray(part)) h.append(el("a", { href: part[1], target: "_blank", rel: "noopener noreferrer", text: part[0] }));
    else h.append(part);
  }
  $("#t-modelhint").textContent = p === "groq"
    ? "Default is a good all-rounder. Reasoning models (gpt-oss, qwen) also work."
    : p === "anthropic" ? "Default: claude-opus-5-5. Use claude-sonnet-5-5 for a cheaper run." : "";
}

async function loadModels() {
  const key = $("#t-key").value.trim(), p = provider();
  const b = $("#t-load-models"); b.disabled = true; b.textContent = "Loading…";
  try {
    const { ok, data } = await post("/api/models", { provider: p, api_key: key });
    if (!ok) throw new Error(data.error || "Could not load models.");
    const dl = $("#t-models"); dl.textContent = "";
    for (const id of data.models) dl.append(el("option", { value: id }));
    // If the current model isn't available to this key, pick the best one that is.
    const m = $("#t-model");
    if (!data.models.includes(m.value)) {
      const pref = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"];
      m.value = pref.find(x => data.models.includes(x)) || data.models[0];
    }
    $("#t-modelhint").textContent = `${data.models.length} models available to your key. Click the box and pick one, or type any id.`;
  } catch (e) { showStatus(e.message, true); }
  finally { b.disabled = false; b.textContent = "Load my models"; }
}

/* ---------- documents ---------- */
function totalChars() { return state.docs.reduce((a, d) => a + d.text.length, 0); }
function addDoc(name, text) {
  text = (text || "").replace(/\u0000/g, "").trim();
  if (!text) { showStatus(`"${name}" has no readable text.`, true); return; }
  if (state.docs.length >= MAX_DOCS) { showStatus(`Max ${MAX_DOCS} documents.`, true); return; }
  if (totalChars() + text.length > MAX_CHARS) { showStatus(`Too much text (limit ${MAX_CHARS.toLocaleString()} characters in total).`, true); return; }
  state.docs.push({ name, text });
  renderDocs();
}
function renderDocs() {
  const ul = $("#t-docs"); ul.textContent = "";
  state.docs.forEach((d, i) => ul.append(el("li", {},
    el("b", { text: d.name, title: d.name }),
    el("span", { text: `${d.text.length.toLocaleString()} chars` }),
    el("button", { type: "button", "aria-label": `Remove ${d.name}`, text: "✕", onclick: () => { state.docs.splice(i, 1); renderDocs(); } }))));
  $("#t-doc-count").textContent = state.docs.length
    ? `${state.docs.length} document(s), ${totalChars().toLocaleString()} characters`
    : "No documents yet.";
}
function loadPdfJs() {
  if (state.pdfReady) return state.pdfReady;
  state.pdfReady = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
    s.onload = () => { window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js"; res(window.pdfjsLib); };
    s.onerror = () => rej(new Error("Could not load the PDF reader (offline?). Paste the text instead."));
    document.head.append(s);
  });
  return state.pdfReady;
}
async function readPdf(file) {
  const lib = await loadPdfJs();
  const pdf = await lib.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const c = await (await pdf.getPage(i)).getTextContent();
    pages.push(c.items.map(x => x.str).join(" "));
  }
  return pages.join("\n\n");
}
async function handleFiles(files) {
  for (const f of files) {
    try {
      showStatus(`Reading ${f.name}…`);
      if (/\.pdf$/i.test(f.name) || f.type === "application/pdf") addDoc(f.name, await readPdf(f));
      else if (f.size > 5e6) showStatus(`${f.name} is too big (5 MB max per file).`, true);
      else addDoc(f.name, await f.text());
    } catch (e) { showStatus(`${f.name}: ${e.message}`, true); }
  }
  if (!$("#t-status").classList.contains("err")) showStatus("");
}

/* ---------- networking ---------- */
async function post(path, body, signal) {
  let r;
  try {
    r = await fetch(API + path, { method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new Error("Could not reach the server. If you opened this page from a file or GitHub Pages, run it from the Vercel site or with `python -m cglc.web`.");
  }
  let data;
  try { data = await r.json(); }
  catch { throw new Error(r.status === 504 || r.status === 502
      ? "The server timed out (hosted runs are limited to about a minute). Try depth “Quick”, a smaller document, or a faster model."
      : `Unexpected server response (${r.status}).`); }
  return { ok: r.ok && data.ok !== false, data };
}
function showStatus(msg, isErr) {
  const s = $("#t-status"); s.textContent = ""; s.classList.toggle("err", !!isErr);
  if (msg && !isErr && /…$|Running/.test(msg)) s.append(el("span", { class: "t-spin" }));
  s.append(msg || "");
}

/* ---------- run ---------- */
let timer = null, abortCtl = null;
const ANSWER_STYLE = "A concise, direct answer to the goal. After each claim, give the supporting document name in square brackets, e.g. [report.pdf].";
function contractMode() { return ($("input[name=t-cmode]:checked") || {}).value || "simple"; }

// The simple form compiled into the full tuple K, so both inputs mean the same thing.
function compileForm() {
  const reqs = $("#t-obl").value.split("\n").map(s => s.trim()).filter(Boolean);
  return {
    goal: $("#t-goal").value.trim(),
    process_duties: $("#t-all").checked
      ? [{ duty_id: "proc-0", description: "Use evidence from every supplied document", check: "use_every_document" }] : [],
    evidence_obligations: reqs,
    soft_prefs: {},
    blockers: [],
    answer_schema: { format: ANSWER_STYLE },
    budget_policy: {}
  };
}
function fillContract() {
  $("#t-contract").value = JSON.stringify(compileForm(), null, 2);
  $("#t-contract-msg").textContent = "Filled from the simple form. Edit anything, then run.";
}
function applyContractMode() {
  const json = contractMode() === "json";
  $("#t-simple").hidden = json; $("#t-jsonwrap").hidden = !json;
  if (json && !$("#t-contract").value.trim()) fillContract();
}

async function run() {
  $("#t-error").hidden = true; $("#t-result").hidden = true;
  const p = provider();
  const jsonMode = contractMode() === "json";
  let contract = null;
  if (jsonMode) {
    try { contract = JSON.parse($("#t-contract").value); }
    catch (e) { return showStatus("The contract is not valid JSON: " + e.message, true); }
    if (!contract || typeof contract !== "object" || Array.isArray(contract)) return showStatus("The contract must be a JSON object.", true);
  }
  const goal = jsonMode ? "" : $("#t-goal").value.trim();
  if (!state.docs.length) { const t = $("#t-paste").value.trim(); if (t) { addDoc("pasted-text.txt", t); $("#t-paste").value = ""; } }
  if (!state.docs.length) return showStatus("Add at least one document first (paste, upload, or load a sample).", true);
  if (!jsonMode && !goal) return showStatus("Type the question or task first.", true);
  if (PROV[p].needsKey && !$("#t-key").value.trim()) return showStatus("Paste your API key, or switch to Offline mode.", true);
  const body = {
    provider: p, api_key: $("#t-key").value.trim(), model: $("#t-model").value.trim(),
    documents: state.docs, depth: $("#t-depth").value, mode: $("#t-mode").value
  };
  if (jsonMode) body.contract = contract;
  else Object.assign(body, { goal, obligations: $("#t-obl").value.split("\n").map(s => s.trim()).filter(Boolean),
                             require_all_docs: $("#t-all").checked });
  const btn = $("#t-run"); btn.disabled = true;
  const t0 = Date.now();
  timer = setInterval(() => showStatus(`Running… ${Math.round((Date.now() - t0) / 1000)}s. The AI works, the controller checks, repeat. Usually 10–60 s; slow free-tier keys can take a few minutes.`), 500);
  showStatus("Running… 0s");
  abortCtl = new AbortController();
  try {
    const { ok, data } = await post("/api/run", body, abortCtl.signal);
    if (!ok) return fail(data);
    state.lastResult = data;
    render(data);
    showStatus(`Done in ${data.spend.elapsed_seconds}s.`);
  } catch (e) {
    if (e.name !== "AbortError") fail({ kind: "net", error: e.message });
  } finally {
    clearInterval(timer); btn.disabled = false;
  }
}
function fail(d) {
  showStatus("");
  const tips = { config: "Check the key and model id.", llm: "Try again, pick another model, or use a smaller document.", input: "" };
  const box = $("#t-error"); box.hidden = false; box.textContent = "";
  box.append(el("b", { text: "That run did not complete." }), d.error || "Unknown error.", tips[d.kind] ? ` ${tips[d.kind]}` : "");
}

/* ---------- explanations ---------- */
const TRIGGER = {
  finalize_request: "the worker asked to finish", lease_expiry: "its work lease ran out",
  structural_stall: "it kept repeating itself and found nothing new", blocker: "it reported a blocker",
  contradiction: "the sources seemed to disagree", budget_tier: "the budget crossed a threshold",
  max_silence: "too many steps passed without a check"
};
const DECISION = {
  ALLOW_FINALIZE: ["Approved", "all gates passed, so the answer may be finalized."],
  CONTINUE: ["Keep going", "the current direction still looks productive; a new bounded lease was issued."],
  VERIFY: ["Double-check", "a claim is weak or disputed; the worker must verify it before anything is approved."],
  REDIRECT: ["Change approach", "the current direction stopped yielding; the worker should try a different angle."],
  ASK_USER: ["Ask the user", "something required can only come from you."],
  REPORT_BLOCKED: ["Stop and report honestly", "the controller could not confirm the answer is ready within the limits, so it will not claim success."]
};
const GATES = [
  ["C_terminal", "Draft exists", "There is a candidate answer to judge."],
  ["C_process", "Required work done", "Any hard duties (e.g. read every document) are complete."],
  ["C_evidence", "Evidence sufficient", "Every requirement has a supporting quote from your documents."],
  ["C_answer", "Answer usable", "The draft is in the requested form."],
  ["C_blocker", "No blocker", "Nothing material is stopping completion."]
];
function gateOk(g, key) { return key === "C_blocker" ? !g[key] : !!g[key]; }

/* ---------- render ---------- */
function render(r) {
  const root = $("#t-result"); root.textContent = ""; root.hidden = false;
  const approved = r.decision === "ALLOW_FINALIZE";
  const offline = r.mode === "offline";
  const lastG = r.final_gates || (r.checkpoints.length ? r.checkpoints[r.checkpoints.length - 1].gates : null);
  const nck = r.checkpoints.length;

  // 1. verdict
  const [dTitle, dMeaning] = DECISION[r.decision] || [r.decision, ""];
  const vclass = approved ? "v-ok" : r.decision === "ASK_USER" ? "v-ask" : "v-stop";
  const timedOut = !approved && r.time_limit_hit;   // stopped by speed/rate limits, not by missing evidence
  const v = el("div", { class: `r-verdict ${timedOut ? "v-ask" : vclass}`, role: "status" },
    el("h3", { text: approved ? "✓ Approved: the controller let this answer through"
      : timedOut ? "⏱ Stopped by the time limit (not by missing evidence)"
      : r.decision === "ASK_USER" ? "Needs your input" : "✕ Not approved: the controller stopped honestly" }),
    el("p", { text: approved
      ? `All five gates passed after ${nck} checkpoint${nck === 1 ? "" : "s"}.`
      : timedOut ? "The run ran out of time (usually the AI provider's rate limit), so no verdict on the evidence was reached. Try depth “Quick”, a faster model (e.g. openai/gpt-oss-20b), a smaller document, or run it locally with python -m cglc.web."
      : `${dTitle}: ${dMeaning} Below is the best draft so far, clearly marked unapproved.` }));
  const why = (timedOut ? [] : (r.reasons || [])).filter(Boolean);
  if (!approved && (why.length || r.blocked_condition)) {
    const ul = el("ul");
    if (r.blocked_condition) ul.append(el("li", { text: r.blocked_condition }));
    why.forEach(x => ul.append(el("li", { text: x })));
    v.append(ul);
  }
  root.append(v);
  (r.warnings || []).forEach(w => root.append(el("div", { class: "r-warn", text: w })));

  // contract: what this run was judged against, and where it came from
  const ct = r.contract || {};
  root.append(el("div", { class: "r-h", text: "The contract this run was judged against" }));
  const cc = el("div", { class: "r-obl" }, el("div", {}, el("b", { text: "Goal: " }), ct.goal || ""));
  (ct.obligations || []).forEach(o => cc.append(el("div", { class: "r-sub", text: "Must be proven: " + o })));
  (ct.duties || []).forEach(d => cc.append(el("div", { class: "r-sub", text: "Hard duty: " + d })));
  cc.append(el("div", { class: "r-sub", text: "Where this contract came from: " + (ct.provenance || "unknown") }));
  (ct.notes || []).forEach(n => cc.append(el("div", { class: "r-sub", text: "• " + n })));
  root.append(cc);

  // 2. answer
  root.append(el("div", { class: "r-h" }, offline ? "Retrieved passages (not an AI answer)" : "Answer",
    el("span", { class: `r-badge ${approved ? "ok" : "stop"}`, text: approved ? "APPROVED" : "UNAPPROVED DRAFT" })));
  root.append(el("p", { class: "r-sub", text: offline ? "Offline mode pastes the best-matching passages. Use an AI provider for a written answer." : "Written by the worker model from your documents. Every claim should map to a quote under “Evidence”." }));
  root.append(el("div", { class: "r-answer", text: r.draft || "(no draft produced)" }));

  // 3. gates
  if (lastG) {
    root.append(el("div", { class: "r-h", text: "The five gates at the last checkpoint" }));
    root.append(el("p", { class: "r-sub", text: "Finalization needs all five. One ✗ and the controller refuses, no matter how good the answer sounds." }));
    const g = el("div", { class: "r-gates" });
    GATES.forEach(([k, name, mean]) => { const ok = gateOk(lastG, k);
      g.append(el("div", { class: `r-gate ${ok ? "pass" : "fail"}` }, el("span", { class: "m", text: ok ? "✓" : "✗" }), el("b", { text: name }), mean)); });
    root.append(g);
  }

  // 4. evidence
  root.append(el("div", { class: "r-h", text: "Evidence for each requirement" }));
  root.append(el("p", { class: "r-sub", text: "A requirement counts as SUPPORTED only when a real quote from your documents backs it. Quotes the AI invented are discarded automatically." }));
  r.evidence.forEach(o => {
    const card = el("div", { class: "r-obl" }, el("div", {},
      el("span", { class: `chip st-${o.status}`, text: o.status }),
      o.contradicted ? el("span", { class: "chip bad", text: "CONTRADICTED" }) : null,
      el("span", { text: o.proposition })));
    if (!o.receipts.length) card.append(el("p", { class: "r-sub", text: "No supporting quote was recorded." }));
    o.receipts.forEach(q => {
      const txt = q.text || ""; const long = txt.length > 420;
      const body = el("span", { text: long ? txt.slice(0, 420) + "…" : txt });
      const bq = el("div", { class: `r-quote ${q.relation === "contradicts" ? "contra" : ""}` },
        el("span", { class: "src", text: `${q.source} · ${q.relation}${q.strength != null ? " · strength " + q.strength : ""}` }), body);
      if (long) { let open = false; bq.append(el("button", { type: "button", text: "show more", onclick: ev => { open = !open; body.textContent = open ? txt : txt.slice(0, 420) + "…"; ev.target.textContent = open ? "show less" : "show more"; } })); }
      card.append(bq);
    });
    root.append(card);
  });

  // 5. timeline
  root.append(el("div", { class: "r-h", text: "What happened, step by step" }));
  root.append(el("p", { class: "r-sub", text: "The worker does the research in short bursts (leases). After each burst the controller inspects the work at a checkpoint and decides what happens next." }));
  const tl = el("div", { class: "r-tl" });
  let si = 0;
  const stepEl = s => el("div", { class: "r-step" },
    el("b", { text: `Worker step ${s.n}` }), ` (${s.intent || "CONTINUE"}) searched `, el("code", { text: s.query || "…" }),
    ` → saw ${s.chunks.length} passage${s.chunks.length === 1 ? "" : "s"}, ${s.new_chunks} new · ${s.tokens.toLocaleString()} tokens`,
    s.dropped ? ` · ${s.dropped} quote${s.dropped === 1 ? "" : "s"} discarded (not found in your documents)` : "",
    s.blocker ? ` · blocker: ${s.blocker}` : "", s.contradiction ? " · reported a contradiction" : "");
  r.checkpoints.forEach(c => {
    while (si < r.steps.length && r.steps[si].n <= c.after_steps) tl.append(stepEl(r.steps[si++]));
    const [t, m] = DECISION[c.decision] || [c.decision, ""];
    const trig = c.triggers.map(x => TRIGGER[x] || x).join("; ");
    const ck = el("div", { class: "r-ck" },
      el("h4", { text: `Checkpoint ${c.id}: ${t}` }),
      el("p", { class: "why", text: `Ran because ${trig}.` }),
      el("p", { text: `Decision: ${m}` }));
    const chips = el("div", { class: "chips" });
    GATES.forEach(([k, name]) => { const ok = gateOk(c.gates, k); chips.append(el("span", { class: `chip ${ok ? "ok" : "bad"}`, text: `${ok ? "✓" : "✗"} ${name}` })); });
    ck.append(chips);
    const reasons = (c.gates.reasons || []).filter(Boolean);
    if (reasons.length) ck.append(el("p", { class: "judge", text: "Why the gate said no: " + reasons.join("; ") }));
    const OVH = /\s*\[overhead guard[^\]]*\]/;
    const heavy = OVH.test(c.note || "");
    const note = (c.note || "").replace(OVH, "").trim();
    if (note) ck.append(el("p", { class: "judge", text: "Judge notes: " + note }));
    if (heavy) ck.append(el("p", { class: "judge", text: "Heads-up: checking cost a lot compared with the work itself, so the overhead guard was triggered." }));
    if (c.rejected && c.rejected.length && c.decision !== "ALLOW_FINALIZE") ck.append(el("p", { class: "judge", text: "Also considered: " + c.rejected.join(", ") }));
    ck.append(el("p", { class: "judge", text: `Controller cost: ${c.controller_tokens.toLocaleString()} tokens` }));
    tl.append(ck);
  });
  while (si < r.steps.length) tl.append(stepEl(r.steps[si++]));
  root.append(tl);

  // 6. spend
  root.append(el("div", { class: "r-h", text: "What it cost" }));
  const sp = r.spend, lim = sp.limits, w = sp.worker;
  const bar = (label, used, max, unit) => el("div", { class: "r-bar" },
    el("div", { text: `${label}: ${Math.round(used).toLocaleString()} of ${Math.round(max).toLocaleString()} ${unit}` }),
    el("div", { class: "track" }, el("div", { class: "fill", style: `width:${Math.min(100, 100 * used / max)}%` })));
  const spend = el("div", { class: "r-spend" },
    bar("Tool calls", w.tool_calls || 0, lim.tool_calls, "allowed"),
    bar("Worker tokens", w.tokens || 0, lim.tokens, "allowed"));
  root.append(spend);
  root.append(el("div", { class: "r-meta" },
    el("span", { class: "chip", text: `mode: ${r.mode === "rag" ? "retrieval (RAG)" : r.mode === "full" ? "full document in prompt" : "offline"}` }),
    r.model ? el("span", { class: "chip", text: `${r.provider}: ${r.model}` }) : null,
    el("span", { class: "chip", text: `controller overhead: ${sp.controller_tokens.toLocaleString()} tokens` }),
    el("span", { class: "chip", text: `${r.docs.count} docs · ${r.docs.chars.toLocaleString()} chars` }),
    el("span", { class: "chip", text: `${sp.elapsed_seconds}s` })));

  // 7. export
  const dl = el("button", { type: "button", text: "Download audit JSON", onclick: () => {
    const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], { type: "application/json" })), download: "cglc-audit.json" });
    document.body.append(a); a.click(); a.remove(); } });
  const raw = el("button", { type: "button", text: "Show raw audit", onclick: ev => {
    const box = $("#t-raw", root);
    if (box) { box.remove(); ev.target.textContent = "Show raw audit"; return; }
    root.append(el("pre", { id: "t-raw", text: JSON.stringify(r, null, 2) })); ev.target.textContent = "Hide raw audit"; } });
  root.append(el("div", { class: "t-run", style: "margin-top:16px" }, dl, raw));
  root.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- wire up ---------- */
function init() {
  if (!$("#try")) return;
  document.querySelectorAll("input[name=t-prov]").forEach(i => i.addEventListener("change", applyProvider));
  $("#t-model").addEventListener("input", e => { e.target.dataset.touched = "1"; });
  $("#t-load-models").addEventListener("click", loadModels);
  $("#t-file").addEventListener("change", e => { handleFiles([...e.target.files]); e.target.value = ""; });
  $("#t-add-paste").addEventListener("click", () => {
    const t = $("#t-paste").value; if (!t.trim()) return;
    addDoc(`pasted-text-${state.docs.length + 1}.txt`, t); $("#t-paste").value = ""; });
  const drop = $("#t-drop");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => handleFiles([...e.dataTransfer.files]));
  document.querySelectorAll("[data-sample]").forEach(b => b.addEventListener("click", () => {
    const s = SAMPLES[b.dataset.sample]; state.docs = []; s.docs.forEach(d => addDoc(d.name, d.text));
    $("#t-goal").value = s.goal; showStatus("");
    if (contractMode() === "json") fillContract(); }));
  $("#t-clear").addEventListener("click", () => { state.docs = []; renderDocs(); });
  $("#t-run").addEventListener("click", run);
  if (!$("#t-obl").value.trim()) $("#t-obl").value = DEFAULT_REQ;
  document.querySelectorAll("input[name=t-cmode]").forEach(i => i.addEventListener("change", applyContractMode));
  $("#t-contract-fill").addEventListener("click", fillContract);
  applyProvider(); renderDocs();
  fetch(API + "/api/health").then(r => r.json()).then(() => {}).catch(() => {
    $("#t-offline-note").hidden = false; });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
