"""Small dependency-free BM25 retriever over document chunks.

Used when a corpus is too large to place in the model's prompt (RAG mode)
and by the offline demo worker. Chunk ids are ``<doc>#c<i>``, the same unit
the CGDP Unique Passage Rate counts.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .worker.llm_worker import chunk_document

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with which what who how why when where do does did "
    "not no yes can could should would may might than then there their they "
    "them these those you your we our i he she his her".split())


def tokenize(text: str) -> List[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    text: str


class BM25Index:
    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks: List[Chunk] = []
        for sid, text in docs.items():
            for i, c in enumerate(chunk_document(text)):
                self.chunks.append(Chunk(f"{sid}#c{i}", sid, c))
        self._tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._by_id = {c.chunk_id: c for c in self.chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def search(self, query: str, k: int = 6,
               exclude: Iterable[str] = ()) -> List[Chunk]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        skip: Set[str] = set(exclude)
        scored = []
        for i, c in enumerate(self.chunks):
            if c.chunk_id in skip:
                continue
            tf, dl, s = self._tf[i], self._len[i], 0.0
            for t in set(q):
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self._avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.chunks[i] for _s, i in scored[:k]]
