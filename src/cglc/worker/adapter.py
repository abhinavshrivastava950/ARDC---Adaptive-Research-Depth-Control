"""Trace adapter + finalization interceptor (Sec 10.1 step 2).

Wraps any DocumentWorker: logs every action/observation as append-only
trace events and forces finalization proposals through the controller
(worker may propose completion, never self-authorize it).

Action restrictions (Sec 7.2): workers enforce the lease's allowed action
classes at their own tool boundary; the adapter does not trust that. It reads
the class the worker reports (``detail["action_class"]``), stores it on the
trace event, and when that class is not in a non-empty allowed list it marks
the event ``status="lease_violation"`` and counts it (``violations``,
``violation_log``). A violation is never silently dropped; the runner books
it on the lease and in the audit record.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..trace import Trace
from ..leases import action_permitted, truncate_observation
from .base import DocumentWorker, WorkerResult


class WorkerAdapter(DocumentWorker):
    def __init__(self, inner: DocumentWorker, trace: Trace, tau_len: int = 3000) -> None:
        self.inner = inner
        self.trace = trace
        self.tau_len = tau_len
        self.violations = 0  # actions whose reported class the lease did not allow
        self.violation_log: List[Dict[str, Any]] = []

    @property
    def controller_note(self) -> str:
        return getattr(self.inner, "controller_note", "")

    @controller_note.setter
    def controller_note(self, value: str) -> None:
        self.inner.controller_note = value

    def act(self, intent, target_gaps, allowed_classes, draft: str) -> WorkerResult:
        res = self.inner.act(intent, target_gaps, allowed_classes, draft)
        chunks: List[str] = []
        for o in res.observations:
            txt, _cut = truncate_observation(o.text, self.tau_len)
            o.text = txt
            chunks.append(o.chunk_id)
        # A worker may report its own query as the action fingerprint (J_t)
        # and a call-level cost that must be charged even with zero cites.
        action_text = (res.detail.get("action_text")
                       or f"{intent} :: {' | '.join(target_gaps)}")
        tot = {"tool_calls": 0.0, "tokens": 0.0, "wall_clock": 0.0}
        for o in res.observations:
            for k, v in o.cost.items():
                tot[k] = tot.get(k, 0.0) + v
        for k, v in (res.detail.get("cost") or {}).items():
            tot[k] = tot.get(k, 0.0) + v
        # Independent check of the lease's action restriction (Sec 7.2).
        action_class = str(res.detail.get("action_class") or "")
        violation = bool(action_class) and not action_permitted(allowed_classes, action_class)
        detail: Dict[str, Any] = {
            "intent": intent, "blocker": res.blocker,
            "contradiction": res.contradiction,
            "action_class": action_class,
            "cited_docs": res.detail.get("cited_docs", []),
            "dropped_citations": res.detail.get("dropped_citations", 0)}
        if violation:
            detail["allowed_classes"] = list(allowed_classes or [])
            detail["propose_final"] = bool(res.propose_final)
        ev = self.trace.log(
            "WORK",
            action_text,
            observation_ids=[o.span_id for o in res.observations],
            chunk_ids=chunks,
            cost=tot,
            status=("lease_violation" if violation
                    else "final_proposal" if res.propose_final else "ok"),
            detail=detail,
        )
        if violation:
            self.violations += 1
            self.violation_log.append({
                "event_id": ev.event_id, "intent": intent,
                "action_class": action_class,
                "allowed_classes": list(allowed_classes or [])})
        return res

    def intercept_finalization(self, draft: str) -> None:
        self.trace.log("FINALIZE_REQUEST", draft[:500], status="final_proposal")
