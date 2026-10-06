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


# --- strictness inside duties and obligations, round trip, weights, examples --------------------------------
FULL = {**BASE,
        "process_duties": [{"check": "cite_document", "document": "policy.txt", "duty_id": "cite"},
                           {"check": "min_distinct_sources", "n": 2},
                           {"check": "use_every_document"}],
        "evidence_obligations": ["a", {"proposition": "b", "required_receipts": 2, "weight": 2.5},
                                 {"proposition": "c", "conditional": True}],
        "soft_prefs": {"style": "concise"}, "blockers": ["the policy file is missing"],
        "clarification_triggers": ["Ask if the country is unclear"],
        "conduct_rules": ["Cite documents by name"], "answer_schema": {"format": "one line"},
        "budget_policy": {"max_tool_calls": 5}, "revision": 3, "contract_id": "policy-q1"}


def test_to_dict_round_trips_through_from_dict_for_every_duty_kind():
    c = TaskContract.from_dict(FULL)
    d = c.to_dict()
    assert d["process_duties"][0]["params"] == {"document": "policy.txt"}
    assert d["process_duties"][1]["params"] == {"n": 2}
    again = TaskContract.from_dict(d)
    assert again.content_hash() == c.content_hash() and again.to_dict() == {**d, "provenance": again.provenance}
    assert again.contract_id == "policy-q1" and again.revision == 3


def test_every_meaningful_field_is_in_to_dict_and_changes_the_hash():
    import dataclasses
    base = TaskContract.from_dict(FULL)
    h = base.content_hash()
    assert h == TaskContract.from_dict(FULL).content_hash()           # stable across loads
    assert h == TaskContract.from_dict({**FULL, "contract_id": "other"}).content_hash()   # the id is not content
    changes = {"goal": "other goal", "blockers": ["something else"], "clarification_triggers": [],
               "conduct_rules": [], "soft_prefs": {}, "answer_schema": {}, "budget_policy": {"max_tool_calls": 6},
               "revision": 4, "process_duties": FULL["process_duties"][:2],
               "evidence_obligations": ["a", {"proposition": "b", "required_receipts": 2, "weight": 3},
                                        {"proposition": "c", "conditional": True}]}
    for k, v in changes.items():
        assert TaskContract.from_dict({**FULL, k: v}).content_hash() != h, k
    assert {f.name for f in dataclasses.fields(TaskContract)} <= set(base.to_dict())


@pytest.mark.parametrize("bad, fragment", [
    ({**BASE, "evidence_obligations": [{"proposition": "a", "wieght": 5}]}, "unknown field"),
    ({**BASE, "evidence_obligations": [{"proposition": "a", "required_recepits": 3}]}, "required_recepits"),
    ({**BASE, "process_duties": [{"check": "use_every_document", "descripton": "typo"}]}, "unknown field"),
    ({**BASE, "process_duties": [{"check": "use_every_document", "n": 2}]}, "does not take"),
    ({**BASE, "process_duties": [{"check": "cite_document", "document": "a", "params": {"document": "b"}}]}, "twice"),
    ({**BASE, "process_duties": [{"check": "min_distinct_sources", "params": {"m": 2}}]}, "does not take"),
    ({**BASE, "process_duties": [{"check": "cite_document", "params": "x"}]}, "params must be"),
    ({**BASE, "process_duties": [{"check": "use_every_document", "duty_id": ""}]}, "duty_id"),
    ({**BASE, "process_duties": [{"check": "use_every_document", "description": ""}]}, "description"),
    ({**BASE, "evidence_obligations": [{"proposition": "a", "obligation_id": None}]}, "obligation_id"),
    ({**BASE, "evidence_obligations": [{"proposition": "a", "weight": 1001}]}, "weight"),
    ({**BASE, "evidence_obligations": ["Same thing.", " same   THING. "]}, "same proposition"),
    ({**BASE, "evidence_obligations": [{"proposition": "a", "conditional": True}]}, "nothing would ever have to be proven"),
    ({**BASE, "contract_id": "bad id!"}, "contract_id"),
    ({**BASE, "contract_id": 5}, "contract_id"),
    ({**BASE, "clarification_triggers": "ask me"}, "clarification_triggers must be a list"),
    ({**BASE, "clarification_triggers": [""]}, "clarification_triggers[0]"),
    ({**BASE, "blockers": ["x y"], "clarification_triggers": ["X  Y"]}, "both a blocker and a clarification trigger"),
    ({**BASE, "conduct_rules": ["r"] * 23}, "too many"),
])
def test_strict_parsing_catches_typos_and_contradictions(bad, fragment):
    with pytest.raises(ValueError, match=fragment.replace("[", r"\[").replace("]", r"\]")):
        TaskContract.from_dict(bad)


def test_conditional_obligations_are_fine_when_something_hard_remains():
    c = TaskContract.from_dict({**BASE, "evidence_obligations": ["a", {"proposition": "b", "conditional": True}]})
    assert c.validate() == []
    d = TaskContract.from_dict({"goal": "g", "process_duties": [{"check": "use_every_document"}],
                                "evidence_obligations": [{"proposition": "b", "conditional": True}]})
    assert d.validate() == []        # a code-checked duty still has to be met


def test_a_supplied_contract_id_is_kept_and_otherwise_a_fresh_one_is_made():
    assert TaskContract.from_dict({**BASE, "contract_id": "mine-1"}).contract_id == "mine-1"
    a, b = TaskContract.from_dict(BASE), TaskContract.from_dict(BASE)
    assert a.contract_id != b.contract_id and a.content_hash() == b.content_hash()


def test_weights_are_validated_and_exposed_as_w_i():
    c = TaskContract.from_dict(FULL)
    assert c.obligation_weights() == {"ev-0": 1.0, "ev-1": 2.5, "ev-2": 1.0}
    for w in (0, -1, float("nan"), float("inf"), True, "2", 1000.5):
        with pytest.raises(ValueError, match="weight"):
            TaskContract.from_dict({**BASE, "evidence_obligations": [{"proposition": "a", "weight": w}]})
    top = TaskContract.from_dict({**BASE, "evidence_obligations": [{"proposition": "a", "weight": 1000}]})
    assert top.obligation_weights() == {"ev-0": 1000.0}


# --- Sec 7.4: validate() on contracts built in code (the runner calls it before the first lease) ---------------
def make(**over):
    c = TaskContract.create("g", None, ["a", "b"])
    for k, v in over.items():
        setattr(c, k, v)
    return c


def test_validate_accepts_a_consistent_contract_and_keeps_the_old_messages():
    assert make().validate() == []
    bad = TaskContract.create("", [], [])
    assert "empty goal" in bad.validate() and any("no hard obligations" in p for p in bad.validate())
    dup = make()
    dup.evidence_obligations[1].obligation_id = "ev-0"
    assert any(p.startswith("duplicate obligation/duty ids") for p in dup.validate())


def test_validate_flags_genuinely_inconsistent_contracts():
    from cglc.contracts import ProcessDuty
    cases = {
        "unknown check": make(process_duties=[ProcessDuty("d", "x", check="vibes")]),
        "cite nothing": make(process_duties=[ProcessDuty("d", "x", check="cite_document", params={"document": " "})]),
        "n zero": make(process_duties=[ProcessDuty("d", "x", check="min_distinct_sources", params={"n": 0})]),
        "n missing": make(process_duties=[ProcessDuty("d", "x", check="min_distinct_sources")]),
        "blank proposition": make(),
        "bad weight": make(),
        "bad receipts": make(),
        "all conditional": make(),
        "blank rule": make(conduct_rules=[" "]),
        "overlap": make(blockers=["x"], clarification_triggers=["X"]),
        "bad budget": make(budget_policy={"max_tokens": -1}),
        "bad stall": make(budget_policy={"structural_stall_parameters": {"jaccard_threshold": 2}}),
        "revision": make(revision=0),
    }
    cases["blank proposition"].evidence_obligations[0].proposition = " "
    cases["bad weight"].evidence_obligations[0].weight = float("nan")
    cases["bad receipts"].evidence_obligations[0].required_receipts = 9
    for o in cases["all conditional"].evidence_obligations:
        o.conditional = True
    for label, c in cases.items():
        assert c.validate(), label
    # duties that the caller verifies with its own function (check == "") stay valid
    assert make(process_duties=[ProcessDuty("d", "read everything")]).validate() == []


def test_the_runner_refuses_an_inconsistent_contract_instead_of_running_it():
    from cglc import Runner
    from cglc.ledger import EvidenceLedger
    from cglc.trace import Trace
    c = TaskContract.create("g", None, ["same", "same"])
    res = Runner().run(c, None, EvidenceLedger([o.obligation_id for o in c.evidence_obligations]), Trace())
    assert res.decision == "ASK_USER" and "same proposition" in res.records[0].note


def test_every_example_contract_file_loads():
    root = Path(__file__).resolve().parents[1] / "examples"
    files = sorted(root.glob("*.json"))
    assert files
    for f in files:
        TaskContract.from_dict(json.loads(f.read_text(encoding="utf-8")))


def test_the_comparison_example_uses_a_checkable_duty_and_keeps_the_rest_as_conduct_rules():
    path = Path(__file__).resolve().parents[1] / "examples" / "comparison_task.json"
    c = TaskContract.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert [(d.duty_id, d.check) for d in c.process_duties] == [("inspect-all", "use_every_document")]
    assert len(c.conduct_rules) == 1 and "Review their diagrams" in c.conduct_rules[0]
    assert c.blockers and c.answer_schema["fields"][0] == "decision"
