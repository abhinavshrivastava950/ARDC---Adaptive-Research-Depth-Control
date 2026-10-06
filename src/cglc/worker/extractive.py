"""Offline extractive worker: no LLM, no key. Shows the control loop only.

Retrieves BM25 passages and pastes them into the draft. It does not
understand the documents; it exists so anyone can watch leases, stall
detection and the finalization gate act on their own text without an API
key. On a REDIRECT lease it skips passages it already returned.
"""
from __future__ import annotations

from typing import Dict, List, Set

from ..leases import SEARCH
from ..retrieval import BM25Index
from .base import DocumentWorker, Observation, WorkerResult
from .llm_worker import EvidenceSpan


class ExtractiveWorker(DocumentWorker):
    def __init__(self, docs: Dict[str, str], query: str, top_k: int = 3) -> None:
        self.index = BM25Index(docs)
        self.query = query
        self.top_k = top_k
        self.seen: Set[str] = set()
        self.evidence: Dict[str, EvidenceSpan] = {}
        self.docs_read: Set[str] = set()

    def act(self, intent: str, target_gaps: List[str],
            allowed_classes: List[str], draft: str) -> WorkerResult:
        self.allowed_classes = list(allowed_classes or [])
        exclude = self.seen if intent == "REDIRECT" else ()
        hits = self.index.search(self.query, self.top_k, exclude)
        fresh = [h for h in hits if h.chunk_id not in self.seen]  # don't repeat text in the draft
        self.seen.update(h.chunk_id for h in hits)
        obs = []
        for h in hits:
            self.docs_read.add(h.source_id)
            self.evidence[h.chunk_id] = EvidenceSpan(h.source_id, h.chunk_id, h.text)
            obs.append(Observation(
                source_id=h.source_id, span_id=h.chunk_id, text=h.text,
                chunk_id=h.chunk_id,
                cost={"tool_calls": 1.0 / max(1, len(hits)),
                      "tokens": float(len(h.text) // 4), "wall_clock": 0.05}))
        new = "\n".join(f"[{h.source_id}] {h.text}" for h in hits)
        merged = (draft + "\n" + new) if draft and new else (new or draft)
        return WorkerResult(
            observations=obs, draft=merged[:6000],
            detail={"action_text": self.query,
                    # BM25 keyword retrieval: the only class this worker has.
                    "action_class": SEARCH,
                    "cost": {} if obs else {"tool_calls": 1.0, "tokens": 0.0,
                                            "wall_clock": 0.05}})
