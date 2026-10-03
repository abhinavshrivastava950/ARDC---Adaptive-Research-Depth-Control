"""Fixed-corpus V1 worker (Sec 10.2 task scope).

Scope: fixed-corpus multi-document comparison, document QA, spec audit.
Retrieval is a transparent TF-IDF ranker (no learned retriever needed
for the first prototype). Execution temperature T_gen=0.7 is exposed for
candidate phrasing diversity; selection stays deterministic.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Dict

from .base import DocumentWorker, WorkerResult, Observation

_WS = re.compile(r"[a-z0-9]+")


def _tok(t: str) -> List[str]:
    return _WS.findall(t.lower())


class FixedCorpusWorker(DocumentWorker):
    def __init__(self, docs: Dict[str, str], top_k: int = 3) -> None:
        self.docs = docs
        self.top_k = top_k
        self._df: Counter = Counter()
        self._doc_tf: Dict[str, Counter] = {}
        for sid, text in docs.items():
            tf = Counter(_tok(text))
            self._doc_tf[sid] = tf
            for term in set(tf):
                self._df[term] += 1
        self.calls = 0

    def _score(self, query: str, sid: str) -> float:
        N = max(1, len(self.docs))
        tf = self._doc_tf[sid]
        s = 0.0
        for term in set(_tok(query)):
            if term in tf:
                idf = math.log(1 + N / (1 + self._df.get(term, 0)))
                s += tf[term] * idf
        return s

    def act(self, intent: str, target_gaps, allowed_classes, draft: str) -> WorkerResult:
        self.calls += 1
        query = " ".join(target_gaps) if target_gaps else (draft or intent)
        ranked = sorted(self.docs, key=lambda sid: -self._score(query or intent, sid))
        obs: List[Observation] = []
        for sid in ranked[: self.top_k]:
            text = self.docs[sid]
            span = text[:1200]
            obs.append(
                Observation(
                    source_id=sid,
                    span_id=f"{sid}#0",
                    text=span,
                    chunk_id=f"{sid}#0",
                    cost={"tool_calls": 1.0, "tokens": float(len(span) // 4),
                          "wall_clock": 0.2},
                )
            )
        new_draft = (draft + "\n" if draft else "") + " ".join(
            o.text[:400] for o in obs
        )[:4000]
        return WorkerResult(observations=obs, draft=new_draft)
