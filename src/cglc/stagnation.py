"""Structural stagnation trigger (Sec 5.2 / 14.2, CGDP f_j0.6_u0.3_p2).

Normative definitions (Sec 14.2):

  H_{t-1} = union_{k<t} C_k
  UPR_t   = |C_t \\ H_{t-1}| / max(1, |C_t|)
  J_t     = max_{i in W_t} Jaccard(tokens(action_t), tokens(action_i))
  Stagnated_t = AND_{k=t-p+1..t} (J_k >= tau_J AND UPR_k <= tau_U)

Stagnation ends the lease and triggers a checkpoint. It never
authorizes finalization and never claims global exhaustion.
"""
from __future__ import annotations

import re
from typing import List, Set

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> Set[str]:
    return set(_TOKEN.findall(text.lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0  # identical no-ops
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def unique_passage_rate(current: List[str], history: Set[str]) -> float:
    """UPR_t: novel-chunk proportion of *current* retrieval (Sec 14.2)."""
    if not current:
        return 0.0
    novel = sum(1 for c in current if c not in history)
    return novel / max(1, len(current))


class StagnationTracker:
    def __init__(self, tau_J: float = 0.6, tau_U: float = 0.3, p: int = 2,
                 recent_window: int = 5) -> None:
        self.tau_J = tau_J
        self.tau_U = tau_U
        self.p = p
        self.recent_window = recent_window
        self._actions: List[str] = []
        self._seen_chunks: Set[str] = set()
        self._rounds: List[tuple[float, float]] = []  # (J_k, UPR_k)

    def step(self, action_text: str, chunk_ids: List[str]) -> tuple[float, float, bool]:
        cur_tok = tokens(action_text)
        window = self._actions[-self.recent_window :]
        if window:
            J_t = max(jaccard(cur_tok, tokens(a)) for a in window)
        else:
            J_t = 0.0
        UPR_t = unique_passage_rate(chunk_ids, self._seen_chunks)
        self._actions.append(action_text)
        for c in chunk_ids:
            self._seen_chunks.add(c)
        self._rounds.append((J_t, UPR_t))
        return J_t, UPR_t, self.stagnated()

    def stagnated(self) -> bool:
        if len(self._rounds) < self.p:
            return False
        tail = self._rounds[-self.p :]
        return all(J >= self.tau_J and U <= self.tau_U for J, U in tail)
