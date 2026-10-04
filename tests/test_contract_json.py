"""The contract K as JSON: parsing, enforcement of each field, and the web entry point."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_service import DOCS, GOOD_KEY, SmartFake  # noqa: E402

from cglc import service  # noqa: E402
from cglc.contracts import TaskContract  # noqa: E402
from cglc.worker.llm_worker import LLMDocumentWorker  # noqa: E402

TWO_DOCS = DOCS  # policy.txt (cited by SmartFake) and hr.md (never cited)


def run(contract, **over):
    req = {"provider": "groq", "api_key": GOOD_KEY, "contract": contract, "documents": TWO_DOCS}
    req.update(over)
    return service.run_task(req, llm_factory=lambda *a, **k: SmartFake())


BASE = {"goal": "How many days can staff work remotely?",
        "evidence_obligations": ["The number of remote days is quoted from the policy."]}


# --- parsing -----------------------------------------------------------------------
def test_from_dict_full_tuple_round_trips():
    c = TaskContract.from_dict({
        **BASE,
        "process_duties": [{"check": "cite_document", "document": "policy.txt"},
                           {"check": "min_distinct_sources", "n": 1, "duty_id": "src"}],
        "evidence_obligations": ["a", {"proposition": "b", "required_receipts": 2, "weight": 2}],
        "soft_prefs": {"style": "concise"}, "blockers": ["policy missing"],
        "answer_schema": {"format": "one line"}, "budget_policy": {"max_tool_calls": 5}})
    d = c.to_dict()
    assert [x["check"] for x in d["process_duties"]] == ["cite_document", "min_distinct_sources"]
    assert d["evidence_obligations"][1]["required_receipts"] == 2
    assert d["soft_prefs"] == {"style": "concise"} and d["blockers"] == ["policy missing"]
    assert d["provenance"] == "user-supplied (JSON contract)"


@pytest.mark.parametrize("bad, fragment", [
    ({"goal": "g", "evidence_obligations": ["a"], "process_duties": ["read everything"]}, "machine-checkable"),
    ({"goal": "g", "evidence_obligations": ["a"], "process_duties": [{"check": "vibes"}]}, "check must be one of"),
    ({"goal": "g", "evidence_obligations": ["a"], "surprise": 1}, "unknown contract field"),
    ({"goal": "g"}, "no hard obligations"),
    ({"evidence_obligations": ["a"]}, "goal"),
    ({"goal": "g", "evidence_obligations": [{"proposition": "a", "required_receipts": 9}]}, "required_receipts"),
    ({"goal": "g", "evidence_obligations": ["a"], "budget_policy": {"max_tokens": -1}}, "max_tokens"),
    ({"goal": "g", "evidence_obligations": ["a"], "blockers": "not a list"}, "blockers must be a list"),
    ("not an object", "JSON object"),
])
def test_from_dict_rejects_bad_input_with_actionable_messages(bad, fragment):
    with pytest.raises(ValueError, match=fragment):
        TaskContract.from_dict(bad)


def test_input_cannot_forge_its_own_provenance():
    c = TaskContract.from_dict({**BASE, "provenance": "benchmark/user-confirmed"})
    assert c.provenance == "user-supplied (JSON contract)"


# --- the web entry point ----------------------------------------------------------------
def test_json_contract_is_used_as_given_and_explained():
    r = run({**BASE, "soft_prefs": {"style": "concise"}, "blockers": ["policy missing"]})
    c = r["contract"]
    assert r["decision"] == "ALLOW_FINALIZE"
    assert c["goal"] == BASE["goal"] and c["provenance"] == "user-supplied (JSON contract)"
    assert c["json"]["soft_prefs"] == {"style": "concise"}
    joined = " ".join(c["notes"])
    assert "not enforced" in joined and "model judgement" in joined


def test_bad_json_contract_is_a_400_with_the_reason():
    status, out = service.handle_run(json.dumps({
        "provider": "groq", "api_key": GOOD_KEY, "documents": TWO_DOCS,
        "contract": {"goal": "g", "evidence_obligations": ["a"], "process_duties": ["read all"]}}).encode())
    assert status == 400 and out["kind"] == "input" and "Contract:" in out["error"]
    assert "machine-checkable" in out["error"]


def test_impossible_duties_are_rejected_before_spending_anything():
    for duty, fragment in [({"check": "cite_document", "document": "ghost.pdf"}, "ghost.pdf"),
                           ({"check": "min_distinct_sources", "n": 5}, "distinct sources")]:
        status, out = service.handle_run(json.dumps({
            "provider": "groq", "api_key": GOOD_KEY, "documents": TWO_DOCS,
            "contract": {**BASE, "process_duties": [duty]}}).encode())
        assert status == 400 and fragment in out["error"]


def test_hard_duties_are_enforced_by_code():
    ok = run({**BASE, "process_duties": [{"check": "cite_document", "document": "policy.txt"}]})
    assert ok["decision"] == "ALLOW_FINALIZE"
    # SmartFake only ever cites policy.txt, so a 2-source duty can never be met
    blocked = run({**BASE, "process_duties": [{"check": "min_distinct_sources", "n": 2}]})
    assert blocked["decision"] != "ALLOW_FINALIZE"
    assert any("process duty" in x for x in blocked["reasons"])


def test_required_receipts_needs_that_many_distinct_spans():
    ok = run({**BASE, "evidence_obligations": [{"proposition": "p", "required_receipts": 1}]})
    assert ok["decision"] == "ALLOW_FINALIZE"
    strict = run({**BASE, "evidence_obligations": [{"proposition": "p", "required_receipts": 2}]})
    assert strict["decision"] != "ALLOW_FINALIZE"
    assert any("supporting spans the contract requires" in c["note"] for c in strict["checkpoints"])


def test_budget_policy_sets_limits_but_never_above_the_server_caps():
    r = run({**BASE, "budget_policy": {"max_tool_calls": 3, "max_tokens": 9_999_999}})
    lim = r["spend"]["limits"]
    assert lim["tool_calls"] == 3 and lim["tokens"] == service.HARD_MAX_TOKENS


def test_soft_prefs_reach_the_worker_and_blockers_reach_the_judge():
    c = TaskContract.from_dict({**BASE, "soft_prefs": {"style": "concise"},
                                "blockers": ["policy missing"]})
    w = LLMDocumentWorker({"a": "text text text"}, c, SmartFake())
    assert "PREFERENCES" in w._task_block("CONTINUE", [], "") and "concise" in w._task_block("CONTINUE", [], "")
    from cglc.judge import LLMJudge
    from cglc.ledger import EvidenceLedger
    j = LLMJudge(SmartFake(), w)
    assert "MATERIAL BLOCKERS" in j._user(c, EvidenceLedger(["ev-0"]), "d")


def test_simple_form_and_json_form_compile_to_the_same_contract_shape():
    simple = service.run_task({"provider": "groq", "api_key": GOOD_KEY, "goal": BASE["goal"],
                               "obligations": BASE["evidence_obligations"], "documents": TWO_DOCS,
                               "require_all_docs": True},
                              llm_factory=lambda *a, **k: SmartFake())["contract"]["json"]
    assert simple["process_duties"][0]["check"] == "use_every_document"
    assert simple["evidence_obligations"][0]["proposition"] == BASE["evidence_obligations"][0]
    assert simple["answer_schema"]["format"]


def test_flask_endpoint_accepts_a_json_contract():
    pytest.importorskip("flask")
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vercel_app2", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.app.test_client().post("/api/run", json={
        "provider": "offline", "documents": TWO_DOCS,
        "contract": {"goal": "remote days per week", "evidence_obligations": ["remote days per week"]}})
    assert r.status_code == 200 and r.json["contract"]["provenance"] == "user-supplied (JSON contract)"
