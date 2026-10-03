"""Trace adapter + finalization interceptor (Sec 10.1 step 2).

Wraps any DocumentWorker: logs every action/observation as append-only
trace events and forces finalization proposals through the controller
(worker may propose completion, never self-authorize it).
"""
from __future__ import annotations

from typing import List

from ..trace import Trace
from ..leases import truncate_observation
from .base import DocumentWorker, WorkerResult


class WorkerAdapter(DocumentWorker):
    def __init__(self, inner: DocumentWorker, trace: Trace, tau_len: int = 3000) -> None:
        self.inner = inner
        self.trace = trace
        self.tau_len = tau_len

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
        self.trace.log(
            "WORK",
            action_text,
            observation_ids=[o.span_id for o in res.observations],
            chunk_ids=chunks,
            cost=tot,
            status="final_proposal" if res.propose_final else "ok",
            detail={"intent": intent, "blocker": res.blocker,
                    "contradiction": res.contradiction,
                    "cited_docs": res.detail.get("cited_docs", [])},
        )
        return res

    def intercept_finalization(self, draft: str) -> None:
        self.trace.log("FINALIZE_REQUEST", draft[:500], status="final_proposal")
