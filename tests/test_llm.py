"""LLM worker/judge/client tests with scripted fakes (no network, no key)."""
from types import SimpleNamespace as NS

import pytest

from cglc import Runner, TaskContract
from cglc.judge import LLMJudge
from cglc.ledger import EvidenceLedger
from cglc.llm import (AnthropicClient, LLMConfigError, LLMError, LLMRefusal,
                      LLMReply, LLMUsage)
from cglc.trace import Trace
from cglc.worker.adapter import WorkerAdapter
from cglc.worker.llm_worker import LLMDocumentWorker, chunk_document

DOCS = {
    "A": "Architecture A uses an always-on semantic coverage layer. It is expensive.",
    "B": "Architecture B keeps a MAP-like research graph. It is controller-heavy.",
}


class FakeLLM:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        self.calls.append(user)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMReply(data=r, usage=LLMUsage(input_tokens=100, output_tokens=50),
                        seconds=0.1)


def contract():
    return TaskContract.create(
        goal="Compare A and B.",
        evidence_obligations=["Cite a document for each architecture's cost."])


def wreply(quotes, draft="d", final=False, **kw):
    return {"query": kw.pop("query", "cost of A"),
            "citations": [{"source_id": s, "quote": q} for s, q in quotes],
            "draft": draft, "propose_final": final,
            "contradiction": False, "blocker": "", **kw}


def jreply(status="SUPPORTED", receipts=(), **kw):
    act = {"progress": "MEDIUM", "verify_weak_claim": 0.0,
           "move_off_stalled_direction": 0.0, "target_open_gap": 1.0,
           "repeat_risk": 0.0}
    base = {
        "obligations": [{"obligation_id": "ev-0", "status": status,
                         "contradicted": False,
                         "receipts": [{"span_id": s, "relation": "supports",
                                       "strength": 0.9, "claim": "c"}
                                      for s in receipts]}],
        "answer_conforms": True, "needs_user": False, "infeasible": False,
        "blocker": "", "has_alternative": False, "direction": "PRODUCTIVE",
        "actions": {"CONTINUE": act, "VERIFY": act, "REDIRECT": act},
        "rationale": "r"}
    base.update(kw)
    return base


def test_chunking_splits_long_paragraph():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))
    assert len(chunk_document(text)) > 1


def test_worker_drops_fabricated_quote_and_charges_cost():
    llm = FakeLLM([wreply([("A", "this quote is not in the corpus at all")])])
    w = LLMDocumentWorker(DOCS, contract(), llm)
    trace = Trace()
    res = WorkerAdapter(w, trace).act("CONTINUE", ["ev-0"], [], "")
    assert res.observations == []
    assert res.detail["dropped_citations"] == 1
    # cost is booked even though nothing was cited
    assert trace.consumed()["tokens"] == 150.0
    assert trace.consumed()["tool_calls"] == 1.0
    # worker's own query is the stall-detector fingerprint
    assert trace.events[-1].action_text == "cost of A"


def test_worker_keeps_verbatim_quote_and_tracks_docs_read():
    llm = FakeLLM([wreply([("A", "always-on semantic coverage layer")])])
    w = LLMDocumentWorker(DOCS, contract(), llm)
    res = w.act("CONTINUE", ["ev-0"], [], "")
    assert [o.span_id for o in res.observations] == ["A#c0"]
    assert w.docs_read == {"A"} and "A#c0" in w.evidence


def test_worker_llm_failure_becomes_blocker_but_config_error_raises():
    w = LLMDocumentWorker(DOCS, contract(), FakeLLM([LLMError("boom")]))
    assert "boom" in w.act("CONTINUE", [], [], "").blocker
    w = LLMDocumentWorker(DOCS, contract(), FakeLLM([LLMRefusal("x")]))
    assert "refused" in w.act("CONTINUE", [], [], "").blocker
    w = LLMDocumentWorker(DOCS, contract(), FakeLLM([LLMConfigError("no key")]))
    with pytest.raises(LLMConfigError):
        w.act("CONTINUE", [], [], "")


def _worker_with_span():
    llm = FakeLLM([wreply([("A", "always-on semantic coverage layer")])])
    w = LLMDocumentWorker(DOCS, contract(), llm)
    w.act("CONTINUE", ["ev-0"], [], "")
    return w


def test_judge_rejects_supported_without_valid_receipt():
    w = _worker_with_span()
    judge = LLMJudge(FakeLLM([jreply("SUPPORTED", receipts=["Z#c9"])]), w)
    j = judge(contract(), EvidenceLedger(["ev-0"]), "draft")
    assert not j.evidence_sufficient and j.open_gaps == ["ev-0"]
    assert "not accepted" in j.rationale


def test_judge_accepts_cited_support_and_fills_ledger():
    w = _worker_with_span()
    judge = LLMJudge(FakeLLM([jreply("SUPPORTED", receipts=["A#c0"])]), w)
    ledger = EvidenceLedger(["ev-0"])
    j = judge(contract(), ledger, "draft")
    assert j.evidence_sufficient and ledger.s["ev-0"] == 1.0
    assert j.receipt_ids and judge.last_tokens == 150.0


def test_judge_fails_closed_on_llm_error():
    w = _worker_with_span()
    judge = LLMJudge(FakeLLM([LLMError("down")]), w)
    j = judge(contract(), EvidenceLedger(["ev-0"]), "draft")
    assert not j.evidence_sufficient and not j.answer_conforms
    assert judge.last_tokens is None


def test_end_to_end_premature_final_is_refused_then_allowed():
    c = contract()
    ledger = EvidenceLedger(["ev-0"])
    trace = Trace()
    llm_w = FakeLLM([
        wreply([], draft="guess", final=True),  # premature, nothing cited
        wreply([("A", "always-on semantic coverage layer")],
               draft="A cited", final=True),
    ])
    w = LLMDocumentWorker(DOCS, c, llm_w)
    llm_j = FakeLLM([jreply("UNSEEN"), jreply("SUPPORTED", receipts=["A#c0"])])
    runner = Runner(judge=LLMJudge(llm_j, w), harvest_receipts=False)
    res = runner.run(c, WorkerAdapter(w, trace), ledger, trace,
                     process_check=lambda: (True, []))
    assert res.decision == "ALLOW_FINALIZE"
    first = [r.decision for r in res.records][:-1]
    assert len(first) == 1 and first[0] in ("CONTINUE", "VERIFY", "REDIRECT")
    cert = res.records[-1].supporting_receipts
    assert cert and cert["ev-0"][0]["span_id"] == "A#c0"
    # controller cost is measured, not the flat 500 estimate
    assert res.records[-1].controller_cost["tokens"] == 150.0


# --- AnthropicClient (SDK injected, no network) ---------------------------
def _resp(text='{"ok": true}', stop="end_turn"):
    return NS(stop_reason=stop, stop_details=NS(category="cyber"),
              content=[NS(type="text", text=text)],
              usage=NS(input_tokens=10, output_tokens=5,
                       cache_creation_input_tokens=3, cache_read_input_tokens=99),
              _request_id="req_1")


def _client(resp):
    sent = {}

    def create(**kw):
        sent.update(kw)
        return resp
    return AnthropicClient(client=NS(messages=NS(create=create))), sent


def test_client_request_shape_has_no_sampling_params_and_default_model(monkeypatch):
    monkeypatch.delenv("CGLC_MODEL", raising=False)
    cl, sent = _client(_resp())
    out = cl.complete_json("sys", "hi", {"type": "object"})
    assert sent["model"] == "claude-opus-5-5"
    assert "temperature" not in sent and "top_p" not in sent
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["effort"] == "medium"
    assert out.data == {"ok": True}
    assert out.usage.work_tokens == 18.0  # cache reads not charged


def test_client_model_env_override(monkeypatch):
    monkeypatch.setenv("CGLC_MODEL", "claude-sonnet-5-5")
    assert AnthropicClient(client=NS(messages=NS())).model == "claude-sonnet-5-5"


def test_client_refusal_and_truncation_and_bad_json():
    with pytest.raises(LLMRefusal):
        _client(_resp(stop="refusal"))[0].complete_json("s", "u", {})
    with pytest.raises(LLMError, match="max_tokens"):
        _client(_resp(stop="max_tokens"))[0].complete_json("s", "u", {})
    with pytest.raises(LLMError, match="JSON"):
        _client(_resp(text="nope"))[0].complete_json("s", "u", {})


def test_missing_credentials_is_a_loud_config_error(monkeypatch):
    def create(**kw):
        raise TypeError("Could not resolve authentication method. Expected ...")
    cl = AnthropicClient(client=NS(messages=NS(create=create)))
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        cl.complete_json("s", "u", {})


def test_key_never_in_repr():
    cl = AnthropicClient(api_key="sk-ant-SECRET", client=NS(messages=NS()))
    assert "SECRET" not in repr(cl)


def test_locate_credits_the_real_document_even_if_model_names_a_passage_id():
    w = LLMDocumentWorker(DOCS, contract(), FakeLLM([]))
    hit = w.locate("B#c0", "keeps a MAP-like research graph")  # wrong id + wrong doc
    assert hit and hit[0] == "B#c0"
    assert w.locate("A", "keeps a MAP-like research graph")[0] == "B#c0"
    assert w.locate("A", "a sentence that is nowhere in the corpus") is None
