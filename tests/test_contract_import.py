"""The cglc-contract-v1 importer: mapping, strictness, and what the controller does with it."""
import copy
import json
from pathlib import Path

import pytest

from cglc import service
from cglc.contracts import TaskContract
from cglc.llm import LLMReply, LLMUsage

FIX = json.loads((Path(__file__).parent / "fixtures" / "cglc_v1_contract.json").read_text(encoding="utf-8"))
DOC = ("Container queries let you style an element based on the size of its container.\n\n"
       "Declare a container with container-type: inline-size, then write an @container rule.\n\n"
       "Browser compatibility: container queries are supported in all modern browsers.")
DOCS = [{"name": "container-queries.txt", "text": DOC}]
KEY = "gsk_" + "a" * 52


# --- mapping -------------------------------------------------------------------------------------
def test_mapping_of_every_part_of_the_format():
    c = TaskContract.from_dict(FIX)
    props = [o.proposition for o in c.evidence_obligations]
    assert c.contract_id == "demo-0001" and c.goal.startswith("In the supplied documentation")
    # 3 evidence obligations + the 3 provable constraints ("must use ...", "must show ..." x2)
    assert len(props) == 6 and props[0] == "Quoted evidence for: target page title/name"
    assert any("must show syntax/example evidence" in p for p in props)
    assert any("must use the supplied documentation" in p for p in props)
    # conduct rules: the duty text and the three "must remain/must not/..." constraints
    assert len(c.conduct_rules) == 3 and c.conduct_rules[0].startswith("Browse only")
    assert not any("must remain" in p or "must not" in p for p in props)
    # conditional wording becomes a conditional obligation
    assert [o.conditional for o in c.evidence_obligations] == [False, False, True, False, False, True]
    assert c.answer_schema == {"format": "short answer; evidence bullets; caveats; no irreversible action"}
    assert c.soft_prefs["persona_id"] == "explicit_constraint_control"
    assert c.blockers == ["Ask or stop if the site requires login or the page is unavailable."]


def test_budget_policy_is_translated_and_provenance_says_the_source_is_only_claimed():
    c = TaskContract.from_dict(FIX)
    bp = c.budget_policy
    assert bp["max_tool_calls"] == 1000000 and bp["max_tokens"] == 100000000 and bp["max_seconds"] == 86400
    assert bp["lease_action_caps"] == {"SHORT": 1, "STANDARD": 2, "EXTENDED": 4}
    assert bp["structural_stall_parameters"]["unique_passage_rate_threshold"] == 0.2
    assert "source claimed by the input" in c.provenance and "not verified" in c.provenance
    assert c.provenance.startswith("user-supplied")


def test_provenance_records_claims_briefly_and_never_checks_the_revision_id():
    long = "x" * 500
    c = TaskContract.from_dict(tweak(provenance={"dataset": long, "task_id": "t\n1"},
                                     revision_id="sha256:" + "a" * 64))
    assert "dataset=" + "x" * 57 + "..." in c.provenance and "task_id=t 1" in c.provenance
    assert "revision_id sha256:" + "a" * 20 + "..., not verified" in c.provenance
    assert "not a sha256 digest" not in c.provenance
    odd = TaskContract.from_dict(tweak(revision_id="v7"))
    assert "revision_id v7, not a sha256 digest, not verified" in odd.provenance
    with pytest.raises(ValueError, match="revision_id must be a string"):
        TaskContract.from_dict(tweak(revision_id=7))
    assert "revision_id" not in TaskContract.from_dict(tweak(revision_id=...)).provenance


def test_the_native_format_without_a_schema_field_still_works():
    c = TaskContract.from_dict({"goal": "g", "evidence_obligations": ["a"]})
    assert c.provenance == "user-supplied (JSON contract)" and c.conduct_rules == []


def test_blockers_are_split_by_wording_into_blockers_and_clarification_triggers():
    asks = ["Ask if a missing budget would materially change the recommendation.",
            "Ask the user if a size is unclear."]
    stops = ["Ask or stop if the site requires login.", "Stop if the page is unavailable.", "The file is missing."]
    c = TaskContract.from_dict(tweak(blockers=asks[:1] + stops[:1] + asks[1:] + stops[1:]))
    assert c.clarification_triggers == asks and c.blockers == stops
    assert c.to_dict()["clarification_triggers"] == asks and c.to_dict()["blockers"] == stops


def test_the_meta_obligation_each_hard_requirement_names_the_obligations_it_covers():
    c = TaskContract.from_dict(tweak(evidence_obligations=["target page title/name",
                                                           "visible evidence for each hard requirement"]))
    meta = c.evidence_obligations[1].proposition
    assert meta.startswith("Quoted evidence for: visible evidence for each hard requirement")
    assert "obligations ev-2, ev-3" in meta                      # the two provable constraints of the fixture
    # no provable constraint -> nothing to name, the text stays as given
    c = TaskContract.from_dict(tweak(evidence_obligations=["visible evidence for each hard requirement"],
                                     hard_task_constraints=["must not buy"]))
    assert c.evidence_obligations[0].proposition == "Quoted evidence for: visible evidence for each hard requirement"


def test_budget_policy_keys_the_docs_call_recorded_are_really_recorded():
    bp = TaskContract.from_dict(FIX).budget_policy
    assert bp["max_controller_calls"] == 100000 and bp["requires_runtime_configuration"] is False
    assert bp["checkpoint_triggers"] == ["worker requests finalization", "lease expires"]
    assert bp["policy_version"] == "demo-policy" and bp["configuration_note"] == "synthetic fixture"


def test_an_imported_contract_round_trips_through_to_dict():
    c = TaskContract.from_dict(FIX)
    again = TaskContract.from_dict(c.to_dict())
    assert again.content_hash() == c.content_hash() and again.contract_id == "demo-0001"


# --- strictness --------------------------------------------------------------------------------------
def tweak(**changes):
    d = copy.deepcopy(FIX)
    for k, v in changes.items():
        if v is ...:
            d.pop(k, None)
        else:
            d[k] = v
    return d


@pytest.mark.parametrize("bad, fragment", [
    (tweak(contract_schema="cglc-contract-v2"), "unknown contract_schema"),
    (tweak(surprise=1), "unknown cglc-contract-v1 field"),
    (tweak(goal=5), "goal must be a non-empty string"),
    (tweak(goal=...), "goal"),
    (tweak(process_duties="browse only"), "list of strings"),
    (tweak(process_duties=[5]), "process_duties[0]"),
    (tweak(hard_task_constraints={"a": 1}), "list of strings"),
    (tweak(evidence_obligations=[], hard_task_constraints=["must not buy things"]), "no evidence obligations"),
    (tweak(soft_preferences=[1]), "soft_preferences must be a JSON object"),
    (tweak(blockers=[None]), "blockers[0]"),
    (tweak(answer_schema=5), "answer_schema"),
    (tweak(contract_id="bad id!"), "contract_id"),
    (tweak(budget_policy={"surprise": 1}), "unknown budget_policy"),
    (tweak(budget_policy={"runtime_budget_caps": {"total_tokens": -1}}), "positive, finite"),
    (tweak(budget_policy={"runtime_budget_caps": {"nope": 1}}), "runtime_budget_caps"),
    (tweak(budget_policy={"structural_stall_parameters": {"jaccard_threshold": 5, "unique_passage_rate_threshold": 0.2, "consecutive_rounds": 2}}), "0 to 1"),
    (tweak(budget_policy={"structural_stall_parameters": {"jaccard_threshold": 0.6}}), "exactly"),
    (tweak(budget_policy={"lease_action_caps": {"SHORT": 3, "STANDARD": 2, "EXTENDED": 4}}), "SHORT <= STANDARD"),
    (tweak(budget_policy={"lease_action_caps": {"SHORT": 1, "STANDARD": 2, "EXTENDED": 99}}), "1 to 9"),
    (tweak(budget_policy={"checkpoint_triggers": "lease expires"}), "checkpoint_triggers"),
    (tweak(budget_policy={"checkpoint_triggers": [5]}), "checkpoint_triggers[0]"),
    (tweak(budget_policy={"requires_runtime_configuration": "no"}), "requires_runtime_configuration"),
    (tweak(evidence_obligations=["same thing", "Same  thing"], hard_task_constraints=[]), "same proposition"),
])
def test_the_importer_rejects_bad_input_with_a_reason(bad, fragment):
    with pytest.raises(ValueError, match=fragment.replace("[", r"\[").replace("]", r"\]")):
        TaskContract.from_dict(bad)


def test_the_web_gate_applies_the_same_strictness_and_never_starts_the_model():
    started = []

    def factory(*a, **k):
        started.append(1)
        raise AssertionError("must not start")

    for bad in (tweak(contract_schema="nope"), tweak(budget_policy={"surprise": 1})):
        st, out = service.handle_run(json.dumps({"provider": "groq", "api_key": KEY, "documents": DOCS,
                                                 "contract": bad}).encode(), llm_factory=factory)
        assert st == 400 and out["kind"] == "input" and not started


# --- what the controller does with it ---------------------------------------------------------------------
class ContractFake:
    """Answers every obligation named in the judge prompt; cites the first span it was given."""
    model = "fake"

    def __init__(self):
        self.prompts = []

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        u = LLMUsage(input_tokens=200, output_tokens=40)
        props = schema["properties"]
        if set(props) == {"queries"}:
            return LLMReply({"queries": ["container queries declare example"]}, u, 0.0)
        if "citations" in props:
            self.prompts.append(("worker", user))
            return LLMReply({"query": "q", "citations": [{"source_id": "container-queries.txt",
                             "quote": "style an element based on the size of its container"}],
                             "draft": "Declare container-type: inline-size and an @container rule. [container-queries.txt]",
                             "propose_final": True, "contradiction": False, "blocker": ""}, u, 0.0)
        self.prompts.append(("judge", user))
        obls = json.loads(user.split("EVIDENCE OBLIGATIONS: ")[1].split("\n")[0])
        span = user.split('"span_id": "')[1].split('"')[0]
        out = []
        for o in obls:
            if o.get("conditional"):
                out.append({"obligation_id": o["obligation_id"], "status": "UNSEEN", "contradicted": False,
                            "not_applicable": True, "receipts": []})
            else:
                out.append({"obligation_id": o["obligation_id"], "status": "SUPPORTED", "contradicted": False,
                            "receipts": [{"span_id": span, "relation": "supports", "strength": 0.9, "claim": "c"}]})
        act = {"progress": "MEDIUM", "verify_weak_claim": 0.0, "move_off_stalled_direction": 0.0,
               "target_open_gap": 1.0, "repeat_risk": 0.0}
        return LLMReply({"obligations": out, "answer_conforms": True, "needs_user": False, "infeasible": False,
                         "blocker": "", "has_alternative": False, "direction": "PRODUCTIVE",
                         "actions": {"CONTINUE": act, "VERIFY": act, "REDIRECT": act}, "rationale": "ok"}, u, 0.0)


def run(contract, fake=None):
    fake = fake or ContractFake()
    req = {"provider": "groq", "api_key": KEY, "contract": contract, "documents": DOCS, "mode": "full"}
    return service.run_task(req, llm_factory=lambda *a, **k: fake), fake


def test_an_imported_contract_runs_end_to_end_and_conditional_obligations_can_be_not_applicable():
    r, fake = run(FIX)
    assert r["decision"] == "ALLOW_FINALIZE"
    status = {e["obligation_id"]: e["status"] for e in r["evidence"]}
    assert status["ev-2"] == "NOT_APPLICABLE" and status["ev-5"] == "NOT_APPLICABLE"
    assert status["ev-0"] == "SUPPORTED"
    assert r["contract"]["contract_id"] == "demo-0001"


def test_the_contracts_controller_settings_are_actually_applied():
    r, _ = run(FIX)
    assert r["config"]["stagnation"]["tau_U"] == 0.2 and r["config"]["stagnation"]["tau_J"] == 0.6
    assert r["config"]["stagnation"]["p"] == 2
    lease = r["config"]["lease"]
    assert (lease["SHORT"], lease["STANDARD"], lease["EXTENDED"]) == (1, 2, 4)
    # the contract asks for a million tool calls; the server caps it
    assert r["spend"]["limits"]["tool_calls"] == service.HARD_MAX_TOOLS
    assert r["spend"]["limits"]["tokens"] == service.HARD_MAX_TOKENS
    notes = " ".join(r["contract"]["notes"])
    assert "NOT machine-checked" in notes and "stall trigger set from the contract" in notes
    assert "lease sizes set from the contract" in notes


def test_conduct_rules_reach_both_the_worker_and_the_judge():
    _, fake = run(FIX)
    worker = [p for who, p in fake.prompts if who == "worker"][0]
    judge = [p for who, p in fake.prompts if who == "judge"][0]
    assert "RULES THE ANSWER MUST FOLLOW" in worker and "Browse only" in worker
    assert "DECLARED RULES" in judge and "must remain within public" in judge


def test_not_applicable_is_ignored_for_obligations_that_are_not_conditional():
    class Cheeky(ContractFake):
        def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
            r = super().complete_json(system, user, schema, max_tokens, temperature)
            if "obligations" in schema["properties"]:
                for o in r.data["obligations"]:
                    o["not_applicable"] = True        # claims N/A on everything, incl. required ones
                    if not o["receipts"]:
                        continue
                    o["status"] = "UNSEEN"
                    o["receipts"] = []
            return r
    r, _ = run(FIX, Cheeky())
    assert r["decision"] != "ALLOW_FINALIZE"              # non-conditional obligations stay unproven


# --- the real WebRider file (a copy lives in tests/fixtures; see test_contract_webrider10.py) --------------
REAL = Path(__file__).parent / "fixtures" / "webrider_first_10_cglc_contracts.json"


def test_all_ten_real_webrider_contracts_import():
    contracts = json.loads(REAL.read_text(encoding="utf-8"))
    assert len(contracts) == 10
    for raw in contracts:
        c = TaskContract.from_dict(raw)
        assert 6 <= len(c.evidence_obligations) <= 8 and len(c.conduct_rules) >= 3
        assert c.budget_policy["structural_stall_parameters"]["unique_passage_rate_threshold"] == 0.2
