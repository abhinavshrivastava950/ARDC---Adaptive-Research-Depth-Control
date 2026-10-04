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
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from .config import BudgetLimits, DEFAULT_CONFIG
from .contracts import TaskContract
from .judge import LLMJudge
from .ledger import EvidenceLedger
from .llm import LLMConfigError, LLMError, clean_api_key, make_llm
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


def _opt_str(req: Dict[str, Any], key: str, default: str = "") -> str:
    """A request field that must be a string if present (never silently str()-ed)."""
    v = req.get(key)
    if v is None:
        return default
    if not isinstance(v, str):
        raise InputError(f"'{key}' must be a string.")
    return v


def _finite(obj: Any) -> Any:
    """Replace NaN/Infinity with null so a response is always strict JSON."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


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
        if d.get("name") is not None and not isinstance(d.get("name"), str):
            raise InputError("A document's 'name' must be a string.")
        if not isinstance(d.get("text"), str):
            raise InputError(f"Document {i + 1}: 'text' must be a string.")
        name = _NAME_BAD.sub("", d.get("name") or f"document-{i + 1}").strip()[:60]
        name = name or f"document-{i + 1}"
        base, n = name, 2
        while name in docs:
            name, n = f"{base}-{n}", n + 1
        text = d["text"].replace("\x00", "").strip()
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
    provider = _opt_str(req, "provider").lower()
    if provider not in PROVIDERS:
        raise InputError(f"Unknown provider (use one of {', '.join(PROVIDERS)}).")
    contract = None
    raw_contract = req.get("contract")
    if "contract" in req:  # present but null/number/list/... is malformed, not "absent"
        # V1 (Sec 4.1): the user supplies the whole tuple K as JSON.
        if not isinstance(raw_contract, dict):
            kind = {bool: "true/false", int: "a number", float: "a number", str: "a string",
                    list: "a list", type(None): "null"}.get(type(raw_contract), "something else")
            raise InputError(f"The contract must be a JSON object, but got {kind}.")
        try:
            contract = TaskContract.from_dict(raw_contract, provenance="user-supplied (JSON contract)")
        except ValueError as e:
            raise InputError(f"Contract: {e}")
        goal = contract.goal
    else:
        goal = _opt_str(req, "goal").strip()
        if not goal:
            raise InputError("Write the question or task you want answered.")
        if len(goal) > MAX_GOAL:
            raise InputError(f"The task is too long (max {MAX_GOAL} characters).")
    model = _opt_str(req, "model").strip() or DEFAULT_MODELS[provider]
    if provider != "offline" and not _MODEL_RE.match(model):
        raise InputError("Invalid model id.")
    raw_obl = req.get("obligations")
    if raw_obl is None:
        raw_obl = []
    if not isinstance(raw_obl, list) or any(not isinstance(o, str) for o in raw_obl):
        raise InputError("'obligations' must be a list of strings.")
    obl = [o.strip()[:MAX_OBL_CHARS] for o in raw_obl if o.strip()][:MAX_OBLIGATIONS]
    require_all = req.get("require_all_docs", False)
    if not isinstance(require_all, bool):
        raise InputError("'require_all_docs' must be true or false.")
    depth = _opt_str(req, "depth") or "standard"
    if depth not in PRESETS:
        raise InputError("Unknown depth preset.")
    mode = _opt_str(req, "mode") or "auto"
    if mode not in ("auto", "full", "rag"):
        raise InputError("Unknown retrieval mode.")
    return dict(provider=provider, goal=goal, model=model, obligations=obl, contract=contract,
                depth=depth, mode=mode, docs=_clean_docs(req.get("documents")),
                require_all_docs=require_all,
                api_key=clean_api_key(_opt_str(req, "api_key")))


# The worker must copy quotes verbatim, so it runs cold. (T_gen 0.7 is for diverse candidate
# generation; at 0.7 the model sometimes paraphrases inside a quote and the quote is discarded.)
WORKER_TEMPERATURE = 0.0
HARD_MAX_TOOLS = 20.0
HARD_MAX_TOKENS = 120_000.0


def _build_contract(c: Dict[str, Any], provider: str, warnings: List[str]) -> TaskContract:
    """The contract for this run. A JSON contract from the user is used as given.
    The simple form (question + requirement lines) is compiled into the same
    structure, so there is one code path and one audit format."""
    if c.get("contract") is not None:
        return c["contract"]
    # Sec 4.1 / 10.1: in V1 the hard obligations come from the user (or the
    # benchmark); the controller never invents them. The page pre-fills a generic
    # requirement that the user sees and can edit. If none arrives at all we fall
    # back to a generic one and say so, never labelling it user-supplied.
    given = c["obligations"]
    if given and given != [DEFAULT_OBLIGATION]:
        obligations, provenance = given, "user-supplied"
    elif given:  # the pre-filled default, shown to the user and left in place
        if provider == "offline":
            obligations = [c["goal"]]
            provenance = "derived from the question (offline demo only)"
        else:
            obligations = given
            provenance = "generic default requirement shown pre-filled and accepted by the user"
    else:
        obligations = [c["goal"] if provider == "offline" else DEFAULT_OBLIGATION]
        provenance = "system default (the user gave no requirements)"
        warnings.append("No requirements were given, so a generic default contract was used. "
                        "The design expects you to state what must be proven; add your own "
                        "requirement lines for a stricter check.")
    duties = ([{"duty_id": "proc-0", "description": "Use evidence from every supplied document",
                "check": "use_every_document"}] if c["require_all_docs"] else [])
    try:
        return TaskContract.from_dict(
            {"goal": c["goal"], "evidence_obligations": obligations, "process_duties": duties,
             "answer_schema": {"format": ANSWER_STYLE}}, provenance=provenance)
    except ValueError as e:
        raise InputError(str(e))


def _check_duties(contract: TaskContract, docs: Dict[str, str]) -> None:
    """Reject duties that can never be met by these documents."""
    for d in contract.process_duties:
        if d.check == "cite_document" and d.params["document"] not in docs:
            raise InputError(f"Duty '{d.duty_id}' needs a document named '{d.params['document']}'; "
                             f"the documents are: {sorted(docs)}")
        if d.check == "min_distinct_sources" and d.params["n"] > len(docs):
            raise InputError(f"Duty '{d.duty_id}' needs {d.params['n']} distinct sources but only "
                             f"{len(docs)} document(s) were given.")


def _process_check(contract: TaskContract, worker, docs: Dict[str, str]):
    """H_proc: hard duties are checked by code, never by the model (Sec 7.2)."""
    if not contract.process_duties:
        return None

    def check():
        read = worker.docs_read
        unmet = []
        for d in contract.process_duties:
            ok = {"use_every_document": set(docs) <= read,
                  "cite_document": d.params.get("document") in read,
                  "min_distinct_sources": len(read) >= d.params.get("n", 1)}.get(d.check, False)
            if not ok:
                unmet.append(d.duty_id)
        return (not unmet, unmet)
    return check


def contract_notes(contract: TaskContract) -> List[str]:
    """Plain statements of what each field of K actually does in this version."""
    notes = ["goal, evidence obligations and answer schema are used by the worker and the judge."]
    if contract.process_duties:
        notes.append("process duties are verified by code: " + "; ".join(
            f"{d.duty_id} ({d.check})" for d in contract.process_duties) + ".")
    if any(o.required_receipts > 1 for o in contract.evidence_obligations):
        notes.append("an obligation with required_receipts = N is accepted only with N distinct supporting spans.")
    if contract.soft_prefs:
        notes.append("soft_prefs are shown to the worker as preferences; they are not enforced.")
    if contract.blockers:
        notes.append("blockers are given to the judge as conditions to watch for; this is a model "
                     "judgement, not a code check.")
    if contract.conduct_rules:
        notes.append("conduct rules are shown to the worker and the judge; NOT machine-checked "
                     "(no code can verify them in a document run): " + "; ".join(contract.conduct_rules))
    if any(o.conditional for o in contract.evidence_obligations):
        notes.append("conditional obligations may be judged not applicable by the judge when their "
                     "own condition does not hold (a model judgement).")
    bp = contract.budget_policy
    if "structural_stall_parameters" in bp:
        s = bp["structural_stall_parameters"]
        notes.append(f"stall trigger set from the contract: tau_J={s['jaccard_threshold']}, "
                     f"tau_U={s['unique_passage_rate_threshold']}, p={s['consecutive_rounds']}.")
    if "lease_action_caps" in bp:
        notes.append(f"lease sizes set from the contract: {bp['lease_action_caps']}.")
    if contract.budget_policy:
        notes.append("budget_policy max_tool_calls / max_tokens / max_seconds set this run's limits "
                     "(capped by the server); other keys are recorded only.")
    if any(o.weight != 1.0 for o in contract.evidence_obligations):
        notes.append("obligation weights are recorded but do not change the decision yet.")
    return notes


# -- deadline wrappers (hosted functions have a hard wall-clock cap) -----------
class _DeadlineWorker(DocumentWorker):
    def __init__(self, inner: DocumentWorker, deadline: float) -> None:
        self.inner, self.deadline = inner, deadline
        self.hit = False

    def act(self, intent, target_gaps, allowed_classes, draft) -> WorkerResult:
        if time.time() > self.deadline:
            self.hit = True
            return WorkerResult(observations=[], draft=draft,
                                blocker="time limit for this demo run reached",
                                detail={"cost": {"tool_calls": 0.0}})
        return self.inner.act(intent, target_gaps, allowed_classes, draft)

    @property
    def controller_note(self) -> str:
        return getattr(self.inner, "controller_note", "")

    @controller_note.setter
    def controller_note(self, value: str) -> None:
        self.inner.controller_note = value

    def __getattr__(self, k):  # evidence / docs_read of the wrapped worker
        return getattr(self.inner, k)


class _DeadlineJudge:
    def __init__(self, inner, deadline: float) -> None:
        self.inner, self.deadline = inner, deadline
        self._late = False
        self.hit = False

    @property
    def last_tokens(self):
        # The rule judge makes no model call, so the controller spends nothing.
        if self._late or self.inner is rule_judge:
            return 0.0
        return getattr(self.inner, "last_tokens", None)

    def __call__(self, contract, ledger, draft) -> Judgment:
        self._late = time.time() > self.deadline
        if self._late:
            self.hit = True
            return Judgment(infeasible=True, blocker_present=True,
                            blocker_reason="time limit for this run reached (not an evidence failure)",
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
                  else os.environ.get("CGLC_TIME_LIMIT", "240"))
    deadline = started + limit
    total_chars = sum(len(t) for t in docs.values())
    warnings: List[str] = []
    contract = _build_contract(c, provider, warnings)
    _check_duties(contract, docs)
    if not contract.answer_schema:
        contract.answer_schema = {"format": ANSWER_STYLE}
    # B_policy: numeric max_* keys set this run's limits, never above the server caps.
    tools, tokens, max_ck = PRESETS[c["depth"]]
    bp = contract.budget_policy
    tools = min(float(bp.get("max_tool_calls", tools)), HARD_MAX_TOOLS)
    tokens = min(float(bp.get("max_tokens", tokens)), HARD_MAX_TOKENS)
    if "max_seconds" in bp:
        limit = max(5.0, min(limit, float(bp["max_seconds"])))
        deadline = started + limit
    cfg = dataclasses.replace(
        DEFAULT_CONFIG, limits=BudgetLimits(tool_calls=float(tools), tokens=float(tokens),
                                            wall_clock=limit * 2))
    sp = bp.get("structural_stall_parameters")
    if sp:  # CGDP stall trigger settings from the contract
        cfg = dataclasses.replace(cfg, stagnation=dataclasses.replace(
            cfg.stagnation, tau_J=float(sp["jaccard_threshold"]),
            tau_U=float(sp["unique_passage_rate_threshold"]), p=int(sp["consecutive_rounds"])))
    lc = bp.get("lease_action_caps")
    if lc:
        cfg = dataclasses.replace(cfg, lease=dataclasses.replace(
            cfg.lease, SHORT=lc["SHORT"], STANDARD=lc["STANDARD"], EXTENDED=lc["EXTENDED"]))
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
        try:
            llm.deadline = deadline  # lets the client fail fast instead of sleeping past the cap
        except Exception:
            pass
        use_rag = (c["mode"] == "rag"
                   or (c["mode"] == "auto" and total_chars > FULL_CONTEXT_CHARS[provider]))
        mode = "rag" if use_rag else "full"
        t_gen, t_sel = WORKER_TEMPERATURE, cfg.temp.T_select
        if use_rag:
            inner = RAGDocumentWorker(docs, contract, llm, top_k=6, temperature=t_gen)
            warnings.append("Large input: retrieval (RAG) mode. Each step reads only the "
                            "top passages for a generated search query.")
        else:
            inner = LLMDocumentWorker(docs, contract, llm, temperature=t_gen)
        judge = LLMJudge(llm, inner, temperature=t_sel)
        harvest = False

    worker = _DeadlineWorker(inner, deadline)
    process_check = _process_check(contract, inner, docs)

    djudge = _DeadlineJudge(judge, deadline)
    runner = Runner(cfg=cfg, judge=djudge,
                    max_checkpoints=max_ck, harvest_receipts=harvest)
    res = runner.run(contract, WorkerAdapter(worker, trace), ledger, trace,
                     process_check=process_check)
    out = _serialize(c, mode, contract, ledger, trace, inner, res, cfg, warnings,
                     time.time() - started, len(docs), total_chars)
    na = getattr(judge, "na", set())
    for e in out["evidence"]:
        if e["obligation_id"] in na:   # the judge's verdict at the last checkpoint
            e["status"] = "NOT_APPLICABLE"
    out["contract"]["json"] = contract.to_dict()
    out["contract"]["notes"] = contract_notes(contract)
    out["time_limit_hit"] = bool(worker.hit or djudge.hit
                                 or getattr(locals().get("llm"), "deadline_hit", False))
    if out["time_limit_hit"] and out["decision"] != "ALLOW_FINALIZE":
        out["blocked_condition"] = (f"time limit reached (hosted runs are capped at ~{limit:.0f}s); "
                                    "this is a speed/rate-limit stop, not proof the evidence is missing")
    return out


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
            "contradiction": bool(e.detail.get("contradiction")),
            "dropped": int(e.detail.get("dropped_citations", 0))})
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
        "contract": {"goal": contract.goal, "provenance": contract.provenance,
                     "contract_id": contract.contract_id, "conduct_rules": contract.conduct_rules,
                     "duties": [d.description for d in contract.process_duties],
                     "obligations": [o.proposition for o in contract.evidence_obligations]},
        "docs": {"count": n_docs, "chars": total_chars,
                 "names": sorted(c["docs"])},
        "spend": {"worker": {k: round(v, 2) for k, v in used.items()},
                  "controller_tokens": round(ctrl),
                  "limits": cfg.limits.as_dict(), "elapsed_seconds": round(elapsed, 1)},
        "config": {"stagnation": dataclasses.asdict(cfg.stagnation),
                   "lease": dataclasses.asdict(cfg.lease),
                   "temperature": {"worker": WORKER_TEMPERATURE, "judge": cfg.temp.T_select},
                   "depth": c["depth"]},
        "warnings": warnings,
    }


# -- HTTP-agnostic handlers --------------------------------------------------------
def _fail(status: int, kind: str, msg: str, *secrets) -> Tuple[int, Dict[str, Any]]:
    return status, {"ok": False, "kind": kind, "error": scrub(msg, *secrets)}


def _parse(raw: bytes) -> Dict[str, Any]:
    if len(raw) > MAX_BODY_BYTES:
        raise InputError("Request too large.")
    def no_constants(name: str):
        raise ValueError(name)  # NaN / Infinity / -Infinity are not valid JSON
    try:
        d = json.loads(raw.decode("utf-8"), parse_constant=no_constants)
    except RecursionError:
        raise InputError("The JSON is nested too deeply.")
    except (ValueError, UnicodeDecodeError):
        raise InputError("Request body must be valid JSON (NaN and Infinity are not allowed).")
    if not isinstance(d, dict):
        raise InputError("Request body must be a JSON object.")
    return d


def _unexpected(e: Exception, key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """500 for a bug: log a key-scrubbed traceback (visible in the host's logs) and
    tell the user the exception type and where, never its message."""
    try:
        sys.stderr.write(scrub("".join(traceback.format_exception(e)), key) + "\n")
        where = traceback.extract_tb(e.__traceback__)[-1].name
    except Exception:
        where = "unknown"
    return _fail(500, "server", f"Unexpected error ({type(e).__name__} in {where}). "
                                "If this repeats, please report it.", key)


def handle_run(raw: bytes, **kw) -> Tuple[int, Dict[str, Any]]:
    key = None
    try:
        req = _parse(raw)
        key = str(req.get("api_key") or "") or None
        return 200, _finite(run_task(req, **kw))
    except InputError as e:
        return _fail(400, "input", str(e), key)
    except LLMConfigError as e:
        return _fail(400, "config", str(e), key)
    except LLMError as e:
        return _fail(502, "llm", str(e), key)
    except Exception as e:  # never leak internals or the key
        return _unexpected(e, key)


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
        return _unexpected(e, key)


def handle_health() -> Tuple[int, Dict[str, Any]]:
    return 200, {"ok": True, "providers": list(PROVIDERS),
                 "limits": {"max_docs": MAX_DOCS, "max_total_chars": MAX_TOTAL_CHARS}}
