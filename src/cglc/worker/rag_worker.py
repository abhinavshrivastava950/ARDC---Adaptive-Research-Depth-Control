"""RAG document worker: for corpora too large to put in the prompt.

One action = (1) the model writes a keyword query for the lease's intent,
(2) BM25 retrieves passages, (3) the model reads only those passages,
cites verbatim quotes and revises the draft. Citations are grounded
exactly as in ``LLMDocumentWorker`` (a quote must exist in the document).

Retrieval novelty is real here: Observations are the *retrieved* chunks,
so the CGDP Unique Passage Rate measures whether retrieval is still
turning up passages the worker has not seen. On a REDIRECT lease the
worker excludes already-retrieved chunks; the controller only names the
direction, the worker decides how to realize it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from ..contracts import TaskContract
from ..llm import LLMClient, LLMConfigError, LLMError, system_blocks
from ..retrieval import BM25Index
from .base import Observation, WorkerResult
from .llm_worker import LLMDocumentWorker, _norm

QUERY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

QUERY_SYSTEM = """You write search queries for a keyword (BM25) retriever over a set of documents. Given the task, the controller's intent, the open gaps, the current draft and past queries, reply with ONE short keyword-style query (4 to 12 words, no operators) most likely to surface passages that close the gaps. For intent REDIRECT the query must approach the gaps from a different angle than every past query. Use words that would literally appear in the documents."""

RAG_NOTE = """

You are given only RETRIEVED PASSAGES, not the whole corpus. Cite only text that appears in them. source_id must be the passage's source attribute (the document name), not its id."""


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

    def act(self, intent: str, target_gaps: List[str],
            allowed_classes: List[str], draft: str) -> WorkerResult:
        task = self._task_block(intent, target_gaps, draft)
        past = "\n".join(f"- {q}" for q in self.past_queries[-6:]) or "(none)"
        cost1: Dict[str, float] = {"tokens": 0.0, "wall_clock": 0.0}
        try:
            qr = self.llm.complete_json(
                system_blocks(QUERY_SYSTEM), f"{task}\n\nPAST QUERIES:\n{past}",
                QUERY_SCHEMA, 300, self.temperature)
            query = _norm(str(qr.data.get("query", ""))) or self._fallback_query(target_gaps)
            cost1 = {"tokens": qr.usage.work_tokens, "wall_clock": qr.seconds}
        except LLMConfigError:
            raise
        except LLMError:
            query = self._fallback_query(target_gaps)  # retrieval still proceeds
        self.past_queries.append(query)

        exclude = self.seen_chunks if intent == "REDIRECT" else ()
        hits = self.index.search(query, self.top_k, exclude)
        if not hits:
            hits = self.index.search(self._fallback_query(target_gaps), self.top_k, exclude)
        self.seen_chunks.update(h.chunk_id for h in hits)

        passages = "\n\n".join(
            f'<passage source="{h.source_id}" id="{h.chunk_id}">\n{h.text}\n</passage>'
            for h in hits) or "(no passages matched this query)"
        user = f"{task}\n\nSEARCH QUERY USED: {query}\n\nRETRIEVED PASSAGES:\n{passages}"
        res = self._step(self._system, user, draft, prior_cost=cost1,
                         extra_detail={"retrieved": [h.chunk_id for h in hits]})
        res.detail["action_text"] = query  # fingerprint = what was actually searched
        res.observations = [Observation(source_id=h.source_id, span_id=h.chunk_id,
                                        text=h.text, chunk_id=h.chunk_id) for h in hits]
        return res
