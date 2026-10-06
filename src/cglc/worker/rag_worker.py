"""RAG document worker: for corpora too large to put in the prompt.

One SEARCH action = (1) the model writes up to three keyword queries, one per
thing still to be found, (2) the retriever returns passages for each (BM25, or
BM25 + embeddings fused by RRF when a semantic backend is available; see
``retrieval.make_index``), (3) the model reads only those passages, cites
verbatim quotes and revises the draft. Citations are grounded exactly as in
``LLMDocumentWorker`` (a quote must exist in the document).

Retrieval novelty is real here: Observations are the *retrieved* chunks, so
the CGDP Unique Passage Rate measures whether retrieval is still turning up
passages the worker has not seen (chunk ids are the same whichever retriever
runs). On a REDIRECT lease the worker excludes already-retrieved chunks; the
controller only names the direction, the worker decides how to realize it.

Lease enforcement (Sec 7.2, Sec 3.3): the worker enforces the lease's allowed
action classes at its own tool boundary. Without SEARCH it runs no new
retrieval; it re-reads passages it already retrieved or cited (READ, or VERIFY
on a VERIFY lease) and never invents a passage when there is nothing to re-read.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

from ..contracts import TaskContract
from ..embeddings import Embedder
from ..llm import LLMClient, LLMConfigError, LLMError, system_blocks
from ..retrieval import Chunk, make_index
from .base import Observation, WorkerResult
from .llm_worker import LLMDocumentWorker, _norm

MAX_QUERIES = 3

QUERY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}

QUERY_SYSTEM = """You write search queries for a keyword (BM25) retriever over a set of documents. Given the task, the controller's intent, the open gaps, the current draft and past queries, reply with 1 to 3 short keyword-style queries (4 to 12 words each, no operators): ONE PER DISTINCT THING STILL TO BE FOUND, so a question with two parts gets two queries. Use words that would literally appear in the documents, and leave out the subject's own name when it appears everywhere. For intent REDIRECT every query must approach the gaps from a different angle than all past queries."""

HYBRID_QUERY_NOTE = " The retriever also matches by meaning (embeddings), so a synonym or paraphrase of the wording in the documents is fine; keep each query short and specific."

RAG_NOTE = """

You are given only RETRIEVED PASSAGES, not the whole corpus. Cite only text that appears in them. source_id must be the passage's source attribute (the document name), not its id.

The passages are a SAMPLE of the documents. If they do not contain what you need, that does NOT mean the documents lack it: do not set `blocker` for that. Write the best partial draft from what the passages DO support (or leave the draft unchanged) and put in `query` what should be searched for next."""


class RAGDocumentWorker(LLMDocumentWorker):
    def __init__(self, docs: Dict[str, str], contract: TaskContract,
                 llm: LLMClient, top_k: int = 6, max_tokens: int = 6000,
                 temperature: float | None = None,
                 retriever: str = "auto", embedder: Embedder | None = None) -> None:
        super().__init__(docs, contract, llm, max_tokens=max_tokens,
                         temperature=temperature, include_corpus=False)
        self._system = system_blocks(self._system[0]["text"] + RAG_NOTE)
        # Sec 14.7: what was configured and what actually runs is recorded here.
        self.index, self.retrieval_info = make_index(docs, retriever, embedder)
        semantic_hybrid = (self.retrieval_info["used"] == "hybrid"
                           and self.retrieval_info["semantic"])
        self._query_system = QUERY_SYSTEM + (HYBRID_QUERY_NOTE if semantic_hybrid else "")
        self.top_k = top_k
        self.past_queries: List[str] = []
        self.seen_chunks: Set[str] = set()
        self._seen_order: List[str] = []  # first-seen order, for "most recently seen"

    def _mark_seen(self, chunk_ids: Iterable[str]) -> None:
        for cid in chunk_ids:
            if cid not in self.seen_chunks:
                self.seen_chunks.add(cid)
                self._seen_order.append(cid)

    def _fallback_query(self, target_gaps: List[str]) -> str:
        words = " ".join(self._gap_text(g) for g in target_gaps) or self.contract.goal
        return _norm(f"{self.contract.goal} {words}")[:300]

    def _retrieve(self, queries: List[str], exclude) -> List[Chunk]:
        """Top passages for every query, interleaved so each query gets a share."""
        per = max(2, -(-self.top_k // len(queries)))
        lists = [self.index.search(q, per, exclude) for q in queries]
        merged: List[Chunk] = []
        seen: Set[str] = set()
        for rank in range(per):
            for lst in lists:
                if rank < len(lst) and lst[rank].chunk_id not in seen:
                    seen.add(lst[rank].chunk_id)
                    merged.append(lst[rank])
        return merged[: self.top_k]

    def act(self, intent: str, target_gaps: List[str],
            allowed_classes: List[str], draft: str) -> WorkerResult:
        # An empty list means unrestricted (Sec 7.2); the adapter verifies the
        # class reported in detail["action_class"] against this same list.
        self.allowed_classes = list(allowed_classes or [])
        if not self.allowed_classes or "SEARCH" in self.allowed_classes:
            res = self._search(intent, target_gaps, draft)
            res.detail["action_class"] = "SEARCH"
            return res
        return self._reread(intent, target_gaps, draft)

    # -- SEARCH: new retrieval over the corpus ---------------------------
    def _search(self, intent: str, target_gaps: List[str], draft: str) -> WorkerResult:
        task = self._task_block(intent, target_gaps, draft)
        past = "\n".join(f"- {q}" for q in self.past_queries[-6:]) or "(none)"
        cost1: Dict[str, float] = {"tokens": 0.0, "wall_clock": 0.0}
        queries: List[str] = []
        try:
            qr = self.llm.complete_json(
                system_blocks(self._query_system), f"{task}\n\nPAST QUERIES:\n{past}",
                QUERY_SCHEMA, 400, self.temperature)
            queries = [_norm(str(q)) for q in qr.data.get("queries", []) if _norm(str(q))][:MAX_QUERIES]
            cost1 = {"tokens": qr.usage.work_tokens, "wall_clock": qr.seconds}
        except LLMConfigError:
            raise
        except LLMError:
            pass  # retrieval still proceeds with a fallback query
        queries = queries or [self._fallback_query(target_gaps)]
        self.past_queries.extend(queries)

        exclude = self.seen_chunks if intent == "REDIRECT" else ()
        hits = self._retrieve(queries, exclude)
        if not hits:
            hits = self._retrieve([self._fallback_query(target_gaps)], exclude)
        if not self.seen_chunks:
            # First step: also read each document's opening, where definitions live.
            have = {h.chunk_id for h in hits}
            hits = [c for c in self.index.openings(per_doc=1) if c.chunk_id not in have] + hits
        self._mark_seen(h.chunk_id for h in hits)

        passages = "\n\n".join(
            f'<passage source="{h.source_id}" id="{h.chunk_id}">\n{h.text}\n</passage>'
            for h in hits) or "(no passages matched these queries)"
        shown = " | ".join(queries)
        user = f"{task}\n\nSEARCH QUERIES USED: {shown}\n\nRETRIEVED PASSAGES:\n{passages}"
        res = self._step(self._system, user, draft, prior_cost=cost1,
                         extra_detail={"retrieved": [h.chunk_id for h in hits], "queries": queries})
        res.detail["action_text"] = shown  # fingerprint = what was actually searched
        res.observations = [Observation(source_id=h.source_id, span_id=h.chunk_id,
                                        text=h.text, chunk_id=h.chunk_id) for h in hits]
        return res

    # -- no SEARCH: re-read what was already retrieved or cited ----------
    def _reread_class(self, intent: str) -> str:
        allowed = self.allowed_classes
        if "VERIFY" in allowed and (intent == "VERIFY" or "READ" not in allowed):
            return "VERIFY"
        if "READ" in allowed:
            return "READ"
        return "ANSWER"  # neither searching nor re-reading is permitted: revise the draft only

    def _passages_to_reread(self) -> List[Chunk]:
        """At most top_k previously seen passages: those backing verified
        evidence spans first (latest cited first), then the most recently seen."""
        ids = list(reversed(self.evidence)) + list(reversed(self._seen_order))
        out: List[Chunk] = []
        taken: Set[str] = set()
        for cid in ids:
            if cid in taken:
                continue
            ch = self.index.get(cid)
            if ch is None and cid in self.evidence:
                ev = self.evidence[cid]
                ch = Chunk(cid, ev.source_id, ev.text)
            if ch is None:
                continue
            taken.add(cid)
            out.append(ch)
            if len(out) >= self.top_k:
                break
        return out

    def _reread(self, intent: str, target_gaps: List[str], draft: str) -> WorkerResult:
        cls = self._reread_class(intent)
        hits = self._passages_to_reread() if cls != "ANSWER" else []
        if not hits and (cls != "ANSWER" or not draft.strip()):
            # Nothing seen yet (or no draft to revise): make no model call and
            # invent nothing. The action slot is still charged so a lease of
            # no-ops cannot run for free.
            what = ("no passage has been retrieved or cited yet" if cls != "ANSWER"
                    else "there is no draft to revise")
            return WorkerResult(
                observations=[], draft=draft,
                detail={"action_class": cls, "action_text": f"{cls}: nothing to do",
                        "note": f"{cls} step skipped: this lease does not allow new search and {what}.",
                        "cost": {"tool_calls": 1.0, "tokens": 0.0, "wall_clock": 0.0}})
        self._mark_seen(h.chunk_id for h in hits)
        task = self._task_block(intent, target_gaps, draft)
        allowed = ", ".join(self.allowed_classes)
        if hits:
            passages = "\n\n".join(
                f'<passage source="{h.source_id}" id="{h.chunk_id}">\n{h.text}\n</passage>'
                for h in hits)
            body = (f"NO NEW SEARCH IS PERMITTED IN THIS STEP (this lease allows: {allowed}). "
                    f"Work only from the passages below, which were already retrieved or cited. "
                    f"In `query` say what you re-checked.\n\nPASSAGES BEING RE-READ:\n{passages}")
        else:
            body = (f"NO SEARCH AND NO NEW READING IS PERMITTED IN THIS STEP (this lease allows: "
                    f"{allowed}). Revise the draft only; cite nothing you cannot quote verbatim.")
        res = self._step(self._system, f"{task}\n\n{body}", draft,
                         extra_detail={"reread": [h.chunk_id for h in hits]})
        res.detail["action_class"] = cls
        res.detail["action_text"] = f"{cls} " + " ".join(h.chunk_id for h in hits)
        res.observations = [Observation(source_id=h.source_id, span_id=h.chunk_id,
                                        text=h.text, chunk_id=h.chunk_id) for h in hits]
        return res
