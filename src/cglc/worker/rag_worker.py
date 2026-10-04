"""RAG document worker: for corpora too large to put in the prompt.

One action = (1) the model writes up to three keyword queries, one per thing
still to be found, (2) BM25 retrieves passages for each, (3) the model reads
only those passages, cites verbatim quotes and revises the draft. Citations
are grounded exactly as in ``LLMDocumentWorker`` (a quote must exist in the
document).

Retrieval novelty is real here: Observations are the *retrieved* chunks, so
the CGDP Unique Passage Rate measures whether retrieval is still turning up
passages the worker has not seen. On a REDIRECT lease the worker excludes
already-retrieved chunks; the controller only names the direction, the worker
decides how to realize it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from ..contracts import TaskContract
from ..llm import LLMClient, LLMConfigError, LLMError, system_blocks
from ..retrieval import BM25Index, Chunk
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

RAG_NOTE = """

You are given only RETRIEVED PASSAGES, not the whole corpus. Cite only text that appears in them. source_id must be the passage's source attribute (the document name), not its id.

The passages are a SAMPLE of the documents. If they do not contain what you need, that does NOT mean the documents lack it: do not set `blocker` for that. Write the best partial draft from what the passages DO support (or leave the draft unchanged) and put in `query` what should be searched for next."""


class RAGDocumentWorker(LLMDocumentWorker):
    def __init__(self, docs: Dict[str, str], contract: TaskContract,
                 llm: LLMClient, top_k: int = 6, max_tokens: int = 6000,
                 temperature: float | None = None) -> None:
        super().__init__(docs, contract, llm, max_tokens=max_tokens,
                         temperature=temperature, include_corpus=False)
        self._system = system_blocks(self._system[0]["text"] + RAG_NOTE)
        self.index = BM25Index(docs)
        self.top_k = top_k
        self.past_queries: List[str] = []
        self.seen_chunks: Set[str] = set()

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
        task = self._task_block(intent, target_gaps, draft)
        past = "\n".join(f"- {q}" for q in self.past_queries[-6:]) or "(none)"
        cost1: Dict[str, float] = {"tokens": 0.0, "wall_clock": 0.0}
        queries: List[str] = []
        try:
            qr = self.llm.complete_json(
                system_blocks(QUERY_SYSTEM), f"{task}\n\nPAST QUERIES:\n{past}",
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
        self.seen_chunks.update(h.chunk_id for h in hits)

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
