"""Service layer behind the web demo: run CGLC on user-supplied documents.

``run_task`` takes a plain dict (what the browser posts) and returns a plain
dict (what the browser renders). Transport-agnostic: ``api/*.py`` (Vercel)
and ``cglc.web`` (local server) are thin wrappers around ``handle_*``.

Safety: provider and model are validated, sizes are capped, the API key is
used only for the calls of this request, never logged, and scrubbed from any
error text that leaves this module.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import BudgetLimits, DEFAULT_CONFIG
from .contracts import TaskContract
from .judge import LLMJudge
from .ledger import EvidenceLedger
from .llm import LLMConfigError, LLMError, make_llm
from .llm_groq import DEFAULT_GROQ_MODEL, list_groq_models
from .runner import Judgment, Runner, rule_judge
from .trace import Trace
from .worker.adapter import WorkerAdapter
from .worker.base import DocumentWorker, WorkerResult
from .worker.extractive import ExtractiveWorker
from .worker.llm_worker import LLMDocumentWorker
from .worker.rag_worker import RAGDocumentWorker

PROVIDERS = ("groq", "anthropic", "offline")
CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"]
DEFAULT_MODELS = {"groq": DEFAULT_GROQ_MODEL, "anthropic": "claude-opus-5-5", "offline": ""}

MAX_BODY_BYTES = 2_000_000
MAX_DOCS = 20
MAX_TOTAL_CHARS = 600_000
MAX_GOAL = 3000
MAX_OBLIGATIONS = 6
MAX_OBL_CHARS = 400
FULL_CONTEXT_CHARS = {"groq": 8_000, "anthropic": 60_000, "offline": 0}

# depth preset -> (tool_calls, tokens, max_checkpoints)
PRESETS = {"quick": (8, 40_000, 4), "standard": (14, 90_000, 6)}
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,100}$")
_NAME_BAD = re.compile(r"[^A-Za-z0-9 ._\-]")

DEFAULT_OBLIGATION = ("Every claim in the answer is supported by a passage from the "
                      "supplied documents; no claim is left unsupported.")
ANSWER_STYLE = ("A concise, direct answer to the goal. After each claim, give the "
                "supporting document name in square brackets, e.g. [report.pdf].")
_STATUS = {0.0: "UNSEEN", 0.5: "PARTIAL", 1.0: "SUPPORTED"}


class InputError(ValueError):
    pass


def scrub(text: str, *secrets: Optional[str]) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "***")
    return text


# -- request validation ------------------------------------------------------
def _clean_docs(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, list) or not raw:
        raise InputError("Add at least one document (paste text or upload a file).")
    if len(raw) > MAX_DOCS:
        raise InputError(f"Too many documents (max {MAX_DOCS}).")
    docs: Dict[str, str] = {}
    total = 0
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            raise InputError("Malformed document entry.")
        name = _NAME_BAD.sub("", str(d.get("name") or f"document-{i + 1}")).strip()[:60]
        name = name or f"document-{i + 1}"
        base, n = name, 2
        while name in docs:
            name, n = f"{base}-{n}", n + 1
        text = str(d.get("text") or "").replace("\x00", "").strip()
        if not text:
            continue
        total += len(text)
        docs[name] = text
    if not docs:
        raise InputError("The documents are empty.")
    if total > MAX_TOTAL_CHARS:
        raise InputError(f"Documents are too large ({total:,} characters; max {MAX_TOTAL_CHARS:,}).")
    return docs


def _clean_request(req: Dict[str, Any]) -> Dict[str, Any]:
    provider = str(req.get("provider", "")).lower()
    if provider not in PROVIDERS:
        raise InputError(f"Unknown provider (use one of {', '.join(PROVIDERS)}).")
    goal = str(req.get("goal") or "").strip()
    if not goal:
        raise InputError("Write the question or task you want answered.")
    if len(goal) > MAX_GOAL:
        raise InputError(f"The task is too long (max {MAX_GOAL} characters).")
    model = str(req.get("model") or DEFAULT_MODELS[provider]).strip()
    if provider != "offline" and not _MODEL_RE.match(model):
        raise InputError("Invalid model id.")
    obl = [str(o).strip()[:MAX_OBL_CHARS] for o in (req.get("obligations") or [])
           if str(o).strip()][:MAX_OBLIGATIONS]
    depth = str(req.get("depth") or "standard")
    if depth not in PRESETS:
        raise InputError("Unknown depth preset.")
    mode = str(req.get("mode") or "auto")
    if mode not in ("auto", "full", "rag"):
        raise InputError("Unknown retrieval mode.")
    return dict(provider=provider, goal=goal, model=model, obligations=obl,
                depth=depth, mode=mode, docs=_clean_docs(req.get("documents")),
                require_all_docs=bool(req.get("require_all_docs")),
                api_key=str(req.get("api_key") or "").strip() or None)


# -- deadline wrappers (hosted functions have a hard wall-clock cap) -----------
class _DeadlineWorker(DocumentWorker):
    def __init__(self, inner: DocumentWorker, deadline: float) -> None:
        self.inner, self.deadline = inner, deadline

    def act(self, intent, target_gaps, allowed_classes, draft) -> WorkerResult:
        if time.time() > self.deadline:
            return WorkerResult(observations=[], draft=draft,
                                blocker="time limit for this demo run reached",
                                detail={"cost": {"tool_calls": 0.0}})
        return self.inner.act(intent, target_gaps, allowed_classes, draft)

    def __getattr__(self, k):  # evidence / docs_read of the wrapped worker
        return getattr(self.inner, k)


class _DeadlineJudge:
    def __init__(self, inner, deadline: float) -> None:
        self.inner, self.deadline = inner, deadline
        self._late = False

    @property
    def last_tokens(self):
        # The rule judge makes no model call, so the controller spends nothing.
        if self._late or self.inner is rule_judge:
            return 0.0
        return getattr(self.inner, "last_tokens", None)

    def __call__(self, contract, ledger, draft) -> Judgment:
        self._late = time.time() > self.deadline
        if self._late:
            return Judgment(infeasible=True, blocker_present=True,
                            rationale="time limit for this demo run reached",
                            progress_label="UNKNOWN", direction_label="UNKNOWN",
                            open_gaps=[o.obligation_id for o in contract.evidence_obligations])
        return self.inner(contract, ledger, draft)


# -- run -----------------------------------------------------------------------
def run_task(req: Dict[str, Any], time_limit: Optional[float] = None,
             llm_factory=make_llm) -> Dict[str, Any]:
    c = _clean_request(req)
    provider, docs = c["provider"], c["docs"]
    started = time.time()
    limit = float(time_limit if time_limit is not None
                  else os.environ.get("CGLC_TIME_LIMIT", "50"))
    deadline = started + limit
    tools, tokens, max_ck = PRESETS[c["depth"]]
    cfg = dataclasses.replace(
        DEFAULT_CONFIG, limits=BudgetLimits(tool_calls=float(tools), tokens=float(tokens),
                                            wall_clock=limit * 2))
    total_chars = sum(len(t) for t in docs.values())
    warnings: List[str] = []

    obligations = c["obligations"] or [c["goal"] if provider == "offline" else DEFAULT_OBLIGATION]
    contract = TaskContract.create(
        goal=c["goal"],
        process_duties=(["Use evidence from every supplied document"]
                        if c["require_all_docs"] else []),
        evidence_obligations=obligations,
        answer_schema={"format": ANSWER_STYLE},
    )
    trace = Trace()
    ledger = EvidenceLedger([o.obligation_id for o in contract.evidence_obligations])

    if provider == "offline":
        mode = "offline"
        inner: DocumentWorker = ExtractiveWorker(docs, query=c["goal"])
        judge: Any = rule_judge
        harvest = True
        warnings.append("Offline demo: keyword retrieval only, no AI. It shows how the "
                        "controller behaves; it does not understand the documents.")
    else:
        llm = llm_factory(provider, api_key=c["api_key"], model=c["model"])
        use_rag = (c["mode"] == "rag"
                   or (c["mode"] == "auto" and total_chars > FULL_CONTEXT_CHARS[provider]))
        mode = "rag" if use_rag else "full"
        t_gen, t_sel = cfg.temp.T_gen, cfg.temp.T_select
        if use_rag:
            inner = RAGDocumentWorker(docs, contract, llm, top_k=4, temperature=t_gen)
            warnings.append("Large input: retrieval (RAG) mode. Each step reads only the "
                            "top passages for a generated search query.")
        else:
            inner = LLMDocumentWorker(docs, contract, llm, temperature=t_gen)
        judge = LLMJudge(llm, inner, temperature=t_sel)
        harvest = False

    worker = _DeadlineWorker(inner, deadline)
    process_check = None
    if c["require_all_docs"]:
        def process_check():
            unmet = [] if set(docs) <= inner.docs_read else ["proc-0"]
            return (not unmet, unmet)

    runner = Runner(cfg=cfg, judge=_DeadlineJudge(judge, deadline),
                    max_checkpoints=max_ck, harvest_receipts=harvest)
    res = runner.run(contract, WorkerAdapter(worker, trace), ledger, trace,
                     process_check=process_check)
    return _serialize(c, mode, contract, ledger, trace, inner, res, cfg, warnings,
                      time.time() - started, len(docs), total_chars)


def _serialize(c, mode, contract, ledger, trace, worker, res, cfg, warnings,
               elapsed, n_docs, total_chars) -> Dict[str, Any]:
    seen: set = set()
    steps = []
    for e in trace.events:
        if e.action_class != "WORK":
            continue
        new = sum(1 for x in e.chunk_ids if x not in seen)
        seen.update(e.chunk_ids)
        steps.append({
            "n": len(steps) + 1, "intent": e.detail.get("intent", ""),
            "query": e.action_text, "chunks": e.chunk_ids, "new_chunks": new,
            "tokens": round(e.cost.get("tokens", 0.0)), "seconds": round(e.cost.get("wall_clock", 0.0), 2),
            "status": e.status, "blocker": e.detail.get("blocker", ""),
            "contradiction": bool(e.detail.get("contradiction"))})
    checkpoints = [{
        "id": r.checkpoint_id, "triggers": r.trigger_reasons, "decision": r.decision,
        "rejected": r.rejected, "gates": r.gates, "receipts": r.receipt_ids,
        "note": r.note, "controller_tokens": round(r.controller_cost.get("tokens", 0.0)),
        "remaining": {k: round(v, 1) for k, v in r.remaining_budget.items()},
        "after_steps": sum(1 for e in trace.events[:r.trace_len] if e.action_class == "WORK"),
        "blocked_condition": r.blocked_condition} for r in res.records]

    spans = getattr(worker, "evidence", {})
    evidence = []
    for obl in contract.evidence_obligations:
        oid = obl.obligation_id
        evidence.append({
            "obligation_id": oid, "proposition": obl.proposition,
            "status": _STATUS.get(ledger.s.get(oid, 0.0), "PARTIAL"),
            "contradicted": bool(ledger.c.get(oid)),
            "receipts": [{
                "receipt_id": r.receipt_id, "source": r.source_id, "span_id": r.span_id,
                "relation": r.relation, "strength": round(r.strength, 2), "claim": r.proposition,
                "text": spans[r.span_id].text if r.span_id in spans else ""}
                for r in ledger.receipts.get(oid, [])]})

    used = trace.consumed()
    ctrl = sum(r.controller_cost.get("tokens", 0.0) for r in res.records)
    final_gates = res.gates.to_dict() if res.gates else None
    last = res.records[-1] if res.records else None
    reasons = list((final_gates or {}).get("reasons") or (last.gates.get("reasons") if last else []) or [])
    return {
        "ok": True, "mode": mode, "provider": c["provider"],
        "model": c["model"] if c["provider"] != "offline" else "",
        "decision": res.decision, "draft": res.draft,
        "final_gates": final_gates, "reasons": reasons,
        "blocked_condition": last.blocked_condition if last else "",
        "checkpoints": checkpoints, "steps": steps, "evidence": evidence,
        "contract": {"goal": contract.goal,
                     "duties": [d.description for d in contract.process_duties],
                     "obligations": [o.proposition for o in contract.evidence_obligations]},
        "docs": {"count": n_docs, "chars": total_chars,
                 "names": sorted(c["docs"])},
        "spend": {"worker": {k: round(v, 2) for k, v in used.items()},
                  "controller_tokens": round(ctrl),
                  "limits": cfg.limits.as_dict(), "elapsed_seconds": round(elapsed, 1)},
        "config": {"stagnation": dataclasses.asdict(cfg.stagnation),
                   "lease": dataclasses.asdict(cfg.lease),
                   "temperature": {"worker": cfg.temp.T_gen, "judge": cfg.temp.T_select},
                   "depth": c["depth"]},
        "warnings": warnings,
    }


# -- HTTP-agnostic handlers --------------------------------------------------------
def _fail(status: int, kind: str, msg: str, *secrets) -> Tuple[int, Dict[str, Any]]:
    return status, {"ok": False, "kind": kind, "error": scrub(msg, *secrets)}


def _parse(raw: bytes) -> Dict[str, Any]:
    if len(raw) > MAX_BODY_BYTES:
        raise InputError("Request too large.")
    try:
        d = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise InputError("Request body must be JSON.")
    if not isinstance(d, dict):
        raise InputError("Request body must be a JSON object.")
    return d


def handle_run(raw: bytes, **kw) -> Tuple[int, Dict[str, Any]]:
    key = None
    try:
        req = _parse(raw)
        key = str(req.get("api_key") or "") or None
        return 200, run_task(req, **kw)
    except InputError as e:
        return _fail(400, "input", str(e), key)
    except LLMConfigError as e:
        return _fail(400, "config", str(e), key)
    except LLMError as e:
        return _fail(502, "llm", str(e), key)
    except Exception as e:  # never leak internals or the key
        return _fail(500, "server", f"Unexpected error ({type(e).__name__}).", key)


def handle_models(raw: bytes) -> Tuple[int, Dict[str, Any]]:
    key = None
    try:
        req = _parse(raw)
        provider = str(req.get("provider", "")).lower()
        key = str(req.get("api_key") or "") or None
        if provider == "groq":
            if not key:
                raise LLMConfigError("Paste your Groq API key first.")
            models = list_groq_models(key)
            return 200, {"ok": True, "models": models, "default": DEFAULT_GROQ_MODEL}
        if provider == "anthropic":
            return 200, {"ok": True, "models": CLAUDE_MODELS, "default": CLAUDE_MODELS[0]}
        raise InputError("Provider has no model list.")
    except InputError as e:
        return _fail(400, "input", str(e), key)
    except LLMConfigError as e:
        return _fail(400, "config", str(e), key)
    except LLMError as e:
        return _fail(502, "llm", str(e), key)
    except Exception as e:
        return _fail(500, "server", f"Unexpected error ({type(e).__name__}).", key)


def handle_health() -> Tuple[int, Dict[str, Any]]:
    return 200, {"ok": True, "providers": list(PROVIDERS),
                 "limits": {"max_docs": MAX_DOCS, "max_total_chars": MAX_TOTAL_CHARS}}
