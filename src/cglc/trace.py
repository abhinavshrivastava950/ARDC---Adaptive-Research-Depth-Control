"""Append-only raw trace (Sec 4.2, Table 10).

Only the raw trace, contract, active lease and budget accounting are
continuously maintained. Semantic packet S_t is built on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import hashlib


@dataclass
class TraceEvent:
    event_id: int
    time: float
    action_class: str  # e.g. SEARCH / READ / VERIFY / ANSWER / ...
    action_text: str  # raw action / query string (fingerprinted for J_t)
    action_fingerprint: str
    source_id: str = ""
    observation_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    cost: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"  # ok | blocked | contradiction | final_proposal | ...
    detail: Dict[str, Any] = field(default_factory=dict)


def fingerprint(action_text: str) -> str:
    return hashlib.sha256(action_text.encode("utf-8")).hexdigest()[:16]


class Trace:
    """Append-only tau_t store plus running budget consumption."""

    def __init__(self) -> None:
        self.events: List[TraceEvent] = []
        self._next = 0

    def log(
        self,
        action_class: str,
        action_text: str,
        source_id: str = "",
        observation_ids: List[str] | None = None,
        chunk_ids: List[str] | None = None,
        cost: Dict[str, float] | None = None,
        status: str = "ok",
        detail: Dict[str, Any] | None = None,
    ) -> TraceEvent:
        ev = TraceEvent(
            event_id=self._next,
            time=time.time(),
            action_class=action_class,
            action_text=action_text,
            action_fingerprint=fingerprint(action_text),
            source_id=source_id,
            observation_ids=list(observation_ids or []),
            chunk_ids=list(chunk_ids or []),
            cost=dict(cost or {}),
            status=status,
            detail=dict(detail or {}),
        )
        self.events.append(ev)
        self._next += 1
        return ev

    def consumed(self) -> Dict[str, float]:
        tot: Dict[str, float] = {}
        for e in self.events:
            for k, v in e.cost.items():
                tot[k] = tot.get(k, 0.0) + v
        return tot

    def action_texts(self) -> List[str]:
        return [e.action_text for e in self.events]

    def __len__(self) -> int:
        return len(self.events)
