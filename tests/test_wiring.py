"""Wiring of the new lease / progress / retrieval features through the service (no network)."""
import json

import pytest

from cglc import service
from cglc.contracts import TaskContract
from cglc.ledger import EvidenceLedger
from cglc.runner import Runner, rule_judge
from cglc.trace import Trace
from cglc.worker.adapter import WorkerAdapter
from cglc.worker.fixed_corpus import FixedCorpusWorker
from cglc.worker.llm_worker import blocker_text

from test_service import DOCS, SmartFake, req


def _run_rag(**kw):
    f = SmartFake()
    return service.run_task(req(mode="rag", **kw), llm_factory=lambda *a, **k: f)


# -- retrieval choice is validated and reported --------------------------------------------------
def test_retriever_is_validated():
    status, body = service.handle_run(json.dumps(req(retriever="magic")).encode())
    assert status == 400 and body["kind"] == "input" and "retriever" in body["error"].lower()


def test_default_rag_run_reports_what_search_actually_ran():
    r = _run_rag()
    info = r["retrieval"]
    # conftest forces the lexical hashing embedder, which is not semantic, so "auto" must stay on BM25
    assert info["requested"] == "auto" and info["used"] == "bm25" and not info.get("semantic")
    assert r["steps"][0]["action_class"] == "SEARCH"


def test_hybrid_request_is_honoured_and_labelled_lexical_when_no_real_model():
    r = _run_rag(retriever="hybrid")
    info = r["retrieval"]
    assert info["used"] == "hybrid" and info["semantic"] is False
    assert info["note"] and any("Retrieval:" in w for w in r["warnings"])  # the limitation is shown


def test_full_and_offline_modes_say_there_is_no_embedding_search():
    assert service.run_task(req(), llm_factory=lambda *a, **k: SmartFake())["retrieval"]["used"] == "none"
    r = service.run_task({"provider": "offline", "goal": "remote work days", "documents": DOCS})
    assert r["retrieval"]["used"] == "keyword" and "no embeddings" in r["retrieval"]["note"]


# -- lease, progress and contract identity reach the run record -----------------------------------
def test_every_checkpoint_carries_lease_progress_and_effs():
    r = _run_rag()
    ck = r["checkpoints"][0]
    assert ck["lease"]["category"] in ("SHORT", "STANDARD", "EXTENDED")
    assert ck["lease"]["allowed_action_classes"] and "budget_cap" in ck["lease"]
    assert set(ck["lease_progress"]) >= {"met", "targets", "progressed", "eff_support", "eff_resolve"}
    # Sec 14.3: the two quantities are separate numbers
    assert isinstance(ck["eff_support"], float) and isinstance(ck["eff_resolve"], float)
    assert ck["eff_support"] > 0 and ck["lease_progress"]["met"] is True   # the obligation moved to SUPPORTED
    assert ck["next_lease"] is None                                     # terminal decision: no next lease
    assert r["lease_violations"] == 0
    assert r["contract"]["content_hash"].startswith("sha256:")


def test_content_hash_is_the_same_in_the_summary_and_the_audit_record():
    r = _run_rag()
    c = TaskContract.from_dict(r["contract"]["json"])
    assert c.content_hash() == r["contract"]["content_hash"]


def test_contract_notes_describe_blockers_triggers_and_weights():
    c = TaskContract.from_dict({
        "goal": "g", "evidence_obligations": [{"proposition": "p", "weight": 2}],
        "blockers": ["the page requires a login"],
        "clarification_triggers": ["Which region does the user live in?"]})
    notes = " ".join(service.contract_notes(c))
    assert "closes the gate" in notes and "never close the gate" in notes
    assert "Eff_support" in notes


# -- the checkpoint-limit exit must not invent gate results (found by agent C) --------------------
def test_checkpoint_limit_exit_reports_the_real_last_gates():
    c = TaskContract.from_dict({"goal": "needs something absent", "evidence_obligations": ["a missing fact"]})
    tr = Trace()
    led = EvidenceLedger([o.obligation_id for o in c.evidence_obligations])
    w = WorkerAdapter(FixedCorpusWorker({"a": "unrelated text"}), tr)

    res = Runner(judge=rule_judge, max_checkpoints=2).run(c, w, led, tr)  # nothing matches: never satisfied
    assert res.decision == "REPORT_BLOCKED"
    g = res.gates
    assert g.C_process is True            # the contract has no duties; it was wrongly reported unmet before
    assert g.C_evidence is False and g.C_blocker is False
    assert "max checkpoints exhausted" in g.reasons
    assert "checkpoint limit reached (2)" in res.blocked_condition


# -- a worker that writes "None" as its blocker has not reported one -------------------------------
@pytest.mark.parametrize("v", ["", "none", "None", "N/A", "n/a.", " no blocker ", None])
def test_no_blocker_spellings_are_not_blockers(v):
    assert blocker_text(v) == ""


@pytest.mark.parametrize("v", ["page returned 404", "login required", "none of the documents mention it"])
def test_real_blocker_reasons_are_kept(v):
    assert blocker_text(v) == v
