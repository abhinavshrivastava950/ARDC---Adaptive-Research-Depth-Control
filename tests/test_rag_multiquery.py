"""RAG worker: one query per open gap, and a retrieval miss is not a blocker."""
from cglc.contracts import TaskContract
from cglc.llm import LLMReply, LLMUsage
from cglc.worker.rag_worker import QUERY_SYSTEM, RAG_NOTE, RAGDocumentWorker

DOC = "\n\n".join(
    [f"Filler section {i} about nothing in particular, widgets widgets widgets." for i in range(40)]
    + ["The revenue split is 88 percent to the user and 12 percent to the platform.",
       "Advertisers fund campaigns through Razorpay checkout."])


class QueryFake:
    model = "fake"

    def __init__(self, queries):
        self.queries, self.user_prompts = queries, []

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        u = LLMUsage(input_tokens=10, output_tokens=5)
        if set(schema["properties"]) == {"queries"}:
            return LLMReply({"queries": self.queries}, u, 0.0)
        self.user_prompts.append(user)
        return LLMReply({"query": "q", "citations": [], "draft": "", "propose_final": False,
                         "contradiction": False, "blocker": ""}, u, 0.0)


def contract():
    return TaskContract.from_dict({"goal": "split and funding provider?",
                                   "evidence_obligations": ["split quoted", "provider quoted"]})


def test_each_query_gets_a_share_of_the_retrieved_passages():
    fake = QueryFake(["revenue split user platform percent", "advertisers fund campaigns checkout"])
    w = RAGDocumentWorker({"d.txt": DOC}, contract(), fake, top_k=4, retriever="bm25")
    res = w.act("CONTINUE", ["ev-0", "ev-1"], [], "")
    texts = " ".join(o.text for o in res.observations)
    assert "88 percent" in texts and "Razorpay" in texts           # both parts retrieved
    assert res.detail["queries"] == ["revenue split user platform percent",
                                     "advertisers fund campaigns checkout"]
    assert res.detail["action_text"] == " | ".join(res.detail["queries"])


def test_at_most_three_queries_are_used_and_blank_ones_ignored():
    fake = QueryFake(["a split", "  ", "b funding", "c extra", "d too many", "e too many"])
    w = RAGDocumentWorker({"d.txt": DOC}, contract(), fake, top_k=6, retriever="bm25")
    res = w.act("CONTINUE", ["ev-0"], [], "")
    assert res.detail["queries"] == ["a split", "b funding", "c extra"]


def test_a_failed_query_call_falls_back_to_the_gap_text_and_still_retrieves():
    class Broken(QueryFake):
        def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
            from cglc.llm import LLMError
            if set(schema["properties"]) == {"queries"}:
                raise LLMError("rate limited")
            return super().complete_json(system, user, schema, max_tokens, temperature)
    w = RAGDocumentWorker({"d.txt": DOC}, contract(), Broken([]), top_k=4, retriever="bm25")
    res = w.act("CONTINUE", ["ev-0"], [], "")
    assert res.observations and res.detail["queries"]


def test_prompts_tell_the_model_to_split_by_gap_and_not_to_call_a_miss_a_blocker():
    assert "ONE PER DISTINCT THING" in QUERY_SYSTEM
    assert "does NOT mean the documents lack it" in RAG_NOTE and "do not set `blocker`" in RAG_NOTE
