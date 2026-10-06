"""Dependency-free BM25 retriever, plus an optional BM25 + embedding hybrid.

Used when a corpus is too large to place in the model's prompt (RAG mode)
and by the offline demo worker. Chunk ids are ``<doc>#c<i>``, the same unit
the CGDP Unique Passage Rate counts (Sec 14.2). Retrieval belongs to the
worker (Sec 3.3): swapping BM25 for the hybrid changes which chunks come back,
never how they are identified, so the controller is unaffected.

``HybridIndex`` fuses the BM25 ranking with a dense (embedding) ranking by
Reciprocal Rank Fusion. ``make_index`` chooses between the two and returns an
``info`` dict for the run record (Sec 14.7: every configured value is logged).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .embeddings import Embedder, get_embedder
from .worker.llm_worker import chunk_document

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with which what who how why when where do does did "
    "not no yes can could should would may might than then there their they "
    "them these those you your we our i he she his her".split())

RRF_K = 60               # Cormack et al. 2009 constant
DENSE_DEPTH = 50         # candidates taken from each ranking before fusion
MAX_DENSE_CHUNKS = 3000  # above this the corpus is not embedded; BM25 only
RETRIEVERS = ("auto", "bm25", "hybrid")


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

    def openings(self, per_doc: int = 2, max_docs: int = 3) -> List[Chunk]:
        """The first chunks of the first documents. Overview questions ("what is X")
        are usually answered near the start, where a keyword search on the subject's
        own name (which appears everywhere) ranks poorly."""
        out: List[Chunk] = []
        seen_docs: List[str] = []
        for c in self.chunks:
            if c.source_id not in seen_docs:
                if len(seen_docs) >= max_docs:
                    break
                seen_docs.append(c.source_id)
            if sum(1 for o in out if o.source_id == c.source_id) < per_doc:
                out.append(c)
        return out

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def _scored(self, q: List[str], skip: Set[str]) -> List[Tuple[float, int]]:
        """(score, chunk index) for every non-skipped chunk with score > 0,
        best first; ties keep document order."""
        scored: List[Tuple[float, int]] = []
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
        return scored

    def search(self, query: str, k: int = 6,
               exclude: Iterable[str] = ()) -> List[Chunk]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        scored = self._scored(q, set(exclude))
        return [self.chunks[i] for _s, i in scored[:k]]


def rrf_scores(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> Dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank),
    rank starting at 1. A document missing from a ranking contributes nothing."""
    out: Dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            out[i] = out.get(i, 0.0) + 1.0 / (k + rank)
    return out


class HybridIndex(BM25Index):
    """BM25 + dense retrieval, fused by RRF. Same interface as ``BM25Index``.

    Dense candidates must reach ``min_similarity`` (cosine) to enter the fusion,
    the dense analogue of BM25 returning only chunks with score > 0; without it
    the dense side would always contribute k irrelevant "nearest" chunks.
    Ties are broken by BM25 rank, then dense rank, then document order, so the
    result is deterministic. Chunks are embedded once, at construction.
    """

    def __init__(self, docs: Dict[str, str], embedder: Embedder,
                 k1: float = 1.5, b: float = 0.75, rrf_k: int = RRF_K,
                 depth: int = DENSE_DEPTH, min_similarity: float | None = None,
                 bm25: BM25Index | None = None) -> None:
        if bm25 is not None:  # reuse an already-built lexical index (docs unused)
            self.__dict__.update(bm25.__dict__)
        else:
            super().__init__(docs, k1, b)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.depth = depth
        self.min_similarity = (embedder.default_min_similarity
                               if min_similarity is None else float(min_similarity))
        self._vecs = embedder.embed_documents([c.text for c in self.chunks])
        if len(self._vecs) != len(self.chunks):
            raise ValueError("embedder returned a different number of vectors than chunks")

    def _dense_ranked(self, query: str, skip: Set[str]) -> List[Tuple[float, int]]:
        qv = self.embedder.embed_query(query)
        hits: List[Tuple[float, int]] = []
        for i, c in enumerate(self.chunks):
            if c.chunk_id in skip:
                continue
            s = round(self.embedder.similarity(qv, self._vecs[i]), 6)
            if s >= self.min_similarity:
                hits.append((s, i))
        hits.sort(key=lambda x: (-x[0], x[1]))
        return hits[: self.depth]

    def _fuse(self, query: str, skip: Set[str]):
        q = tokenize(query)
        lex = [i for _s, i in self._scored(q, skip)[: self.depth]]
        dense_hits = self._dense_ranked(query, skip)
        dense = [i for _s, i in dense_hits]
        fused = rrf_scores([lex, dense], self.rrf_k)
        lex_rank = {i: r for r, i in enumerate(lex, start=1)}
        dense_rank = {i: r for r, i in enumerate(dense, start=1)}
        inf = float("inf")
        order = sorted(fused, key=lambda i: (-round(fused[i], 12),
                                             lex_rank.get(i, inf), dense_rank.get(i, inf), i))
        return order, fused, lex_rank, dense_rank, dict((i, s) for s, i in dense_hits)

    def search(self, query: str, k: int = 6,
               exclude: Iterable[str] = ()) -> List[Chunk]:
        if k <= 0 or not self.chunks or not tokenize(query):
            return []
        order = self._fuse(query, set(exclude))[0]
        return [self.chunks[i] for i in order[:k]]

    def explain(self, query: str, k: int = 6,
                exclude: Iterable[str] = ()) -> List[Dict[str, Any]]:
        """The top-k with each side's rank and the dense similarity (for audits)."""
        if k <= 0 or not self.chunks or not tokenize(query):
            return []
        order, fused, lex_rank, dense_rank, sims = self._fuse(query, set(exclude))
        return [{"chunk_id": self.chunks[i].chunk_id, "rrf": fused[i],
                 "bm25_rank": lex_rank.get(i), "dense_rank": dense_rank.get(i),
                 "similarity": sims.get(i)} for i in order[:k]]


def make_index(docs: Dict[str, str], retriever: str = "auto",
               embedder: Embedder | None = None, *,
               min_similarity: float | None = None, rrf_k: int = RRF_K,
               depth: int = DENSE_DEPTH, max_dense_chunks: int | None = None,
               k1: float = 1.5, b: float = 0.75) -> Tuple[BM25Index, Dict[str, Any]]:
    """Build the retriever and a JSON-serialisable record of what was built.

    ``retriever``: "bm25" (keyword only), "hybrid" (BM25 + embeddings, with
    whatever embedder is available, even the lexical hashed one), or "auto"
    (hybrid only when the embedder is a real semantic model; the hashed
    fallback is lexical and would only dilute BM25, so auto keeps BM25).
    Any fallback is explained in ``info["note"]``; ``info["used"]`` is what
    actually runs. ``dense_*``/``semantic``/fusion fields describe the dense
    side only when it is used, else they are None/False.
    """
    if retriever not in RETRIEVERS:
        raise ValueError(f"retriever must be one of {RETRIEVERS}, got {retriever!r}")
    cap = MAX_DENSE_CHUNKS if max_dense_chunks is None else int(max_dense_chunks)
    bm = BM25Index(docs, k1, b)
    info: Dict[str, Any] = {
        "requested": retriever, "used": "bm25", "dense_backend": None,
        "dense_model": None, "semantic": False, "n_chunks": len(bm),
        "fusion": None, "rrf_k": None, "min_similarity": None,
        "dense_depth": None, "max_dense_chunks": cap,
        "bm25_k1": k1, "bm25_b": b, "note": "",
    }
    if retriever == "bm25":
        info["note"] = "BM25 keyword retrieval (requested)."
        return bm, info
    if len(bm) == 0:
        info["note"] = "No chunks to retrieve from; BM25 (empty) used."
        return bm, info
    if len(bm) > cap:
        info["note"] = (f"{len(bm)} chunks exceeds MAX_DENSE_CHUNKS={cap}: "
                        f"embeddings skipped, BM25 only.")
        return bm, info

    if embedder is None:
        embedder, enote = get_embedder()
    else:
        enote = f"using caller-supplied embedder {embedder.name}."
    if retriever == "auto" and not embedder.semantic:
        info["note"] = (f"{enote} Auto uses hybrid only with a semantic backend; "
                        f"{embedder.name} is lexical and would only dilute BM25, so BM25 only.")
        return bm, info
    try:
        idx = HybridIndex(docs, embedder, k1, b, rrf_k=rrf_k, depth=depth,
                          min_similarity=min_similarity, bm25=bm)
    except Exception as e:  # model/runtime failure while embedding the corpus
        info["note"] = (f"{enote} Dense indexing failed ({type(e).__name__}: {e}); "
                        f"BM25 only.")
        return bm, info

    info.update(used="hybrid", dense_backend=embedder.backend, dense_model=embedder.model,
                semantic=bool(embedder.semantic), fusion="rrf", rrf_k=rrf_k,
                min_similarity=idx.min_similarity, dense_depth=depth)
    info["note"] = enote if embedder.semantic else (
        f"{enote} Hybrid with {embedder.name}: it tolerates typos and word forms but "
        f"does NOT match synonyms or paraphrases.")
    return idx, info
