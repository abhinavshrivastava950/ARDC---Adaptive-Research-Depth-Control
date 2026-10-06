"""The ten WebRider ``cglc-contract-v1`` contracts (tests/fixtures/webrider_first_10_cglc_contracts.json):
what the import keeps, where every constraint lands, that each runs end to end, and how the judge is
told about blockers versus questions for the user (B_mat, Sec 4.1 / 14.6)."""
import json
import re
from pathlib import Path

import pytest

from cglc import service
from cglc.contracts import TaskContract
from cglc.judge import JUDGE_SYSTEM, _blocker_text
from cglc.llm import LLMReply, LLMUsage

RAW = json.loads((Path(__file__).parent / "fixtures" / "webrider_first_10_cglc_contracts.json")
                 .read_text(encoding="utf-8"))
IDS = [c["contract_id"] for c in RAW]
KEY = "gsk_" + "a" * 52
DOCS = [{"name": "brand-page.txt",
         "text": "Brand X shampoo and hair care is listed here. Product category: hair care. Rating 4.5 of 5."},
        {"name": "support-page.txt",
         "text": "Compatibility: works with the 2023 models. An aria-label gives a button an accessible name."}]
QUOTE = "Brand X shampoo and hair care is listed here"

ASK = ("Ask if a missing location, date, size, budget, model, dietary need, or compatibility fact "
       "would materially change the recommendation.")
ACCESS = ("Ask or stop if the site requires login, checkout, booking, messaging, private data, or "
          "unavailable page access.")
# the constraints that land in conduct rules; everything else "must show/be/use/address/distinguish" is provable
CONDUCT_ONLY = {"must remain within public, non-logged-in browsing",
                "must not purchase, book, message, submit private forms, or enter credentials",
                "must not assume a named brand carries shampoo unless verified",
                "must avoid medical claims"}


def load(i):
    return TaskContract.from_dict(RAW[i])


# --- structure ---------------------------------------------------------------------------------------
def test_all_ten_import_and_validate():
    assert len(RAW) == 10 and len(set(IDS)) == 10
    for raw in RAW:
        c = TaskContract.from_dict(raw)
        assert c.contract_id == raw["contract_id"] and c.validate() == []
        assert c.goal == raw["goal"].strip()
        assert c.provenance.startswith("user-supplied") and "not verified" in c.provenance
        assert "dataset=WebRider/WebRider" in c.provenance and raw["contract_id"] in c.provenance
        assert c.process_duties == []          # nothing a document run could verify by code
        assert c.revision == 1


@pytest.mark.parametrize("i", range(10))
def test_counts_conditional_flags_and_where_each_constraint_lands(i):
    raw, c = RAW[i], load(i)
    props = [o.proposition for o in c.evidence_obligations]
    constraints = raw["hard_task_constraints"]
    provable = [t for t in constraints if t not in CONDUCT_ONLY]
    conduct_c = [t for t in constraints if t in CONDUCT_ONLY]
    assert len(provable) in (2, 3) and len(conduct_c) in (2, 3, 4)
    # 4 generic evidence obligations + the provable constraints
    assert len(props) == 4 + len(provable) and len(props) in (6, 7)
    for t in raw["evidence_obligations"]:
        assert any(p.startswith("Quoted evidence for: " + t) for p in props), t
    for t in provable:
        assert any(p.endswith(": " + t) for p in props), t
    # every duty and every conduct-style constraint is a conduct rule, in order, and nothing else is
    assert c.conduct_rules == raw["process_duties"] + conduct_c
    for t in conduct_c + raw["process_duties"]:
        assert not any(t in p for p in props)
    # every input line landed exactly once: nothing is dropped
    assert len(props) + len(c.conduct_rules) == (len(raw["evidence_obligations"]) + len(constraints)
                                                 + len(raw["process_duties"]))
    # "when relevant" and "if visible" are the only conditional wordings in these files
    expected = [bool(re.search(r"when relevant|if visible", t)) for t in raw["evidence_obligations"] + provable]
    assert [o.conditional for o in c.evidence_obligations] == expected
    assert sum(expected) in (1, 2)
    assert all(o.weight == 1.0 and o.required_receipts == 1 for o in c.evidence_obligations)


def test_the_if_visible_constraint_is_conditional_and_the_others_are_not():
    c = TaskContract.from_dict(next(r for r in RAW if r["contract_id"] == "rbc4-02397"))
    cond = {o.obligation_id: o.conditional for o in c.evidence_obligations}
    by_text = {o.proposition: o.conditional for o in c.evidence_obligations}
    assert by_text["The documents show, in a quote, that the answer satisfies: must show browser "
                   "compatibility evidence if visible"] is True
    assert sum(cond.values()) == 2           # that one and the generic "... when relevant"


def test_the_meta_obligation_names_the_requirements_it_covers():
    c = load(0)
    meta = c.evidence_obligations[2]
    assert "each hard requirement" in meta.proposition
    assert "ev-4, ev-5" in meta.proposition       # the two provable constraints of rbc4-01037
    assert "must be shampoo/hair-care" in c.evidence_obligations[4].proposition
    c3 = load(3)                                 # three provable constraints
    assert "ev-4, ev-5, ev-6" in c3.evidence_obligations[2].proposition
    assert all(len(o.proposition) <= 400 for o in c3.evidence_obligations)


def test_the_stall_parameters_lease_caps_and_budget_caps_are_carried():
    for raw in RAW:
        c = TaskContract.from_dict(raw)
        bp = c.budget_policy
        assert bp["structural_stall_parameters"] == {"jaccard_threshold": 0.6,
                                                     "unique_passage_rate_threshold": 0.2,
                                                     "consecutive_rounds": 2}
        assert bp["lease_action_caps"] == {"SHORT": 1, "STANDARD": 3, "EXTENDED": 5}
        caps = raw["budget_policy"]["runtime_budget_caps"]
        assert (bp["max_tool_calls"], bp["max_tokens"], bp["max_seconds"]) == (
            caps["worker_actions"], caps["total_tokens"], caps["runtime_seconds"])
        # keys that used to be dropped although the docs said "recorded": now really recorded
        assert bp["max_controller_calls"] == caps["controller_calls"]
        assert bp["checkpoint_triggers"] == raw["budget_policy"]["checkpoint_triggers"] and len(bp["checkpoint_triggers"]) == 6
        assert bp["requires_runtime_configuration"] is False
        assert bp["policy_version"] == raw["budget_policy"]["policy_version"]
        assert c.answer_schema == {"format": raw["answer_schema"]}
        assert c.soft_prefs == raw["soft_preferences"]


def test_blockers_are_split_by_kind_and_nothing_is_dropped():
    for raw in RAW:
        c = TaskContract.from_dict(raw)
        assert raw["blockers"] == [ASK, ACCESS]
        assert c.clarification_triggers == [ASK]          # only the user can answer it
        assert c.blockers == [ACCESS]                      # an access condition that closes the gate
        assert set(c.blockers) | set(c.clarification_triggers) == set(raw["blockers"])


def test_classification_of_blocker_wording():
    from cglc.contract_import import _ASK, _HALT

    def kind(t):
        return "ask" if _ASK.match(t) and not _HALT.search(t) else "blocker"
    assert kind("Ask if the budget is missing.") == "ask"
    assert kind("Ask the user if a size is missing.") == "ask"
    assert kind("  ask if a model is unclear") == "ask"
    assert kind("Ask or stop if the site requires login.") == "blocker"
    assert kind("Stop if the page is unavailable.") == "blocker"
    assert kind("The page is unavailable.") == "blocker"
    assert kind("Ask if the page is unavailable, then halt.") == "blocker"


def test_to_dict_is_honest_round_trips_and_the_hash_covers_the_new_field():
    for raw in RAW:
        c = TaskContract.from_dict(raw)
        d = c.to_dict()
        assert d["clarification_triggers"] == [ASK] and d["blockers"] == [ACCESS]
        again = TaskContract.from_dict(d)
        assert again.content_hash() == c.content_hash()
        assert again.contract_id == c.contract_id         # the file's id is kept across a round trip
        assert again.to_dict() == {**d, "provenance": again.provenance}
        assert TaskContract.from_dict(raw).content_hash() == c.content_hash()   # stable across loads
    c = load(0)
    h = c.content_hash()
    c.clarification_triggers = []
    assert c.content_hash() != h
    c = load(0)
    c.blockers = []
    assert c.content_hash() != h
    c = load(0)
    c.evidence_obligations[0].weight = 2.0
    assert c.content_hash() != h
    assert len({TaskContract.from_dict(r).content_hash() for r in RAW}) == 10


# --- every contract runs end to end ------------------------------------------------------------------------
DECISIONS = {"ALLOW_FINALIZE", "ASK_USER", "REPORT_BLOCKED"}


@pytest.mark.parametrize("i", range(10))
def test_each_contract_runs_offline_through_the_service_without_crashing(i):
    r = service.run_task({"provider": "offline", "contract": RAW[i], "documents": DOCS})
    assert r["ok"] and r["decision"] in DECISIONS and r["mode"] == "offline"
    cj = r["contract"]
    assert cj["contract_id"] == IDS[i] and cj["provenance"].startswith("user-supplied (cglc-contract-v1")
    assert cj["json"]["clarification_triggers"] == [ASK] and cj["json"]["blockers"] == [ACCESS]
    assert len(cj["json"]["evidence_obligations"]) == len(cj["obligations"]) == len(r["evidence"])
    notes = " ".join(cj["notes"])
    assert "NOT machine-checked" in notes and "stall trigger set from the contract" in notes
    assert "lease sizes set from the contract" in notes
    json.dumps(r, allow_nan=False)
    st = r["config"]["stagnation"]
    assert (st["tau_J"], st["tau_U"], st["p"]) == (0.6, 0.2, 2)
    lease = r["config"]["lease"]
    assert (lease["SHORT"], lease["STANDARD"], lease["EXTENDED"]) == (1, 3, 5)
    assert r["spend"]["limits"]["tool_calls"] == service.HARD_MAX_TOOLS     # the server caps the file's 1,000,000
    assert r["spend"]["limits"]["tokens"] == service.HARD_MAX_TOKENS


class Scripted:
    """Fake worker + judge. ``hook(user_prompt, reply)`` may edit each judge reply; the judge prompts
    are kept so a test can see exactly what the judge was told."""
    model = "fake"

    def __init__(self, hook=None, supported=True):
        self.hook, self.supported = hook, supported
        self.judge_prompts = []
        self.worker_prompts = []

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        u = LLMUsage(input_tokens=200, output_tokens=40)
        props = schema["properties"]
        if set(props) == {"queries"}:
            return LLMReply({"queries": ["shampoo hair care rating"]}, u, 0.0)
        if "citations" in props:
            self.worker_prompts.append(user)
            return LLMReply({"query": "shampoo", "citations": [{"source_id": "brand-page.txt", "quote": QUOTE}],
                             "draft": "Brand X shampoo is listed with a 4.5 rating. [brand-page.txt]",
                             "propose_final": True, "contradiction": False, "blocker": ""}, u, 0.0)
        self.judge_prompts.append(user)
        obls = json.loads(user.split("EVIDENCE OBLIGATIONS: ")[1].split("\n")[0])
        span = user.split('"span_id": "')[1].split('"')[0]
        out = []
        for o in obls:
            if o.get("conditional"):
                out.append({"obligation_id": o["obligation_id"], "status": "UNSEEN", "contradicted": False,
                            "not_applicable": True, "receipts": []})
            elif self.supported:
                out.append({"obligation_id": o["obligation_id"], "status": "SUPPORTED", "contradicted": False,
                            "receipts": [{"span_id": span, "relation": "supports", "strength": 0.9, "claim": "c"}]})
            else:
                out.append({"obligation_id": o["obligation_id"], "status": "UNSEEN", "contradicted": False,
                            "receipts": []})
        act = {"progress": "MEDIUM", "verify_weak_claim": 0.0, "move_off_stalled_direction": 0.0,
               "target_open_gap": 1.0, "repeat_risk": 0.0}
        reply = {"obligations": out, "answer_conforms": True, "needs_user": False, "infeasible": False,
                 "blocker": "", "has_alternative": False, "direction": "PRODUCTIVE",
                 "actions": {"CONTINUE": act, "VERIFY": act, "REDIRECT": act}, "rationale": "ok"}
        if self.hook:
            reply = self.hook(user, reply) or reply
        return LLMReply(reply, u, 0.0)


def run(contract, fake, **extra):
    req = {"provider": "groq", "api_key": KEY, "contract": contract, "documents": DOCS, "mode": "full", **extra}
    return service.run_task(req, llm_factory=lambda *a, **k: fake)


def declared(user, label):
    """The JSON list a prompt line declares (what the judge model actually reads)."""
    m = re.search(rf"^{label}[^\n]*?\): (\[.*\])$", user, re.M)
    return json.loads(m.group(1)) if m else None


@pytest.mark.parametrize("mode", ["full", "rag"])
@pytest.mark.parametrize("i", range(10))
def test_each_contract_runs_with_scripted_llm_and_finalizes_on_ordinary_work(i, mode):
    fake = Scripted()
    r = run(RAW[i], fake, mode=mode)
    assert r["mode"] == mode
    assert r["decision"] == "ALLOW_FINALIZE", (r["reasons"], r["checkpoints"][-1]["gates"])
    status = [e["status"] for e in r["evidence"]]
    assert "NOT_APPLICABLE" in status and status.count("SUPPORTED") == len(status) - status.count("NOT_APPLICABLE")
    assert r["contract"]["notes"]
    # the model's view: the access blocker is a blocker, the budget question is a clarification trigger
    judge = fake.judge_prompts[0]
    assert declared(judge, "MATERIAL BLOCKERS") == [ACCESS]
    assert declared(judge, "USER-CLARIFICATION TRIGGERS") == [ASK]
    blockers_line = next(l for l in judge.splitlines() if l.startswith("MATERIAL BLOCKERS"))
    assert "budget" not in blockers_line
    assert "DECLARED RULES" in judge and "Browse only" in judge


# --- blockers versus questions for the user ---------------------------------------------------------------------
def over_eager(user, reply):
    """A literal-minded judge: treats every 'Ask if ...' string it is shown as a blocker that holds."""
    shown = declared(user, "MATERIAL BLOCKERS") or []
    hits = [b for b in shown if b.startswith("Ask if")]
    if hits:
        reply["blocker"] = hits[0]
    if declared(user, "USER-CLARIFICATION TRIGGERS"):
        reply["needs_user"] = True          # and a literal-minded reading of the trigger list
    return reply


OLD_SHAPE = {"goal": "Find sunscreen under makeup with SPF evidence.",
             "evidence_obligations": ["SPF evidence is quoted."],
             "blockers": [ASK, ACCESS]}    # how the importer used to file the 'Ask if ...' string


def test_before_the_split_a_clarification_string_filed_as_a_blocker_closes_the_gate_forever():
    r = run(OLD_SHAPE, Scripted(over_eager))
    assert r["decision"] == "REPORT_BLOCKED"
    assert all(c["gates"]["C_blocker"] for c in r["checkpoints"])
    assert any("unresolved material blocker" in x for c in r["checkpoints"] for x in c["gates"]["reasons"])


def test_after_the_split_the_same_judge_reading_cannot_hold_an_ordinary_task_back():
    fake = Scripted(over_eager)
    r = run(RAW[2], fake)                    # the sunscreen task, same over-eager judge
    # needs_user alone does not stop a run whose every requirement is proven (Sec 14.6: the gate comes first)
    assert r["decision"] == "ALLOW_FINALIZE" and fake.judge_prompts
    assert not any(c["gates"]["C_blocker"] for c in r["checkpoints"])


def test_a_clarification_trigger_asks_the_user_only_while_requirements_are_unproven():
    def needs(user, reply):
        reply["needs_user"] = True

    r = run(RAW[2], Scripted(needs, supported=False))
    assert r["decision"] == "ASK_USER"       # evidence cannot be completed and the judge says only the user can help
    assert r["checkpoints"][-1]["gates"]["C_evidence"] is False


def test_an_access_blocker_that_really_holds_still_closes_the_gate():
    def login_wall(user, reply):
        reply["blocker"] = "the supplied page is a login wall"

    r = run(RAW[2], Scripted(login_wall))
    assert r["decision"] == "REPORT_BLOCKED"
    assert all(c["gates"]["C_blocker"] for c in r["checkpoints"])
    assert any("unresolved material blocker" in x for c in r["checkpoints"] for x in c["gates"]["reasons"])


@pytest.mark.parametrize("none_like", ["none", "None", "N/A", "n/a", " no blocker. ", "Not applicable", "-", "null"])
def test_a_blocker_written_as_none_is_not_a_blocker(none_like):
    def none(user, reply):
        reply["blocker"] = none_like

    assert _blocker_text(none_like) == ""
    r = run(RAW[0], Scripted(none))
    assert r["decision"] == "ALLOW_FINALIZE"


def test_real_blocker_text_is_kept_and_nonsense_lookalikes_are_not_swallowed():
    for t in ("the page requires login", "no access to the page", "none of the documents mention SPF",
              "N/A page unavailable", "404", "403 forbidden"):
        assert _blocker_text(t) == t
    assert _blocker_text(None) == "" and _blocker_text("") == "" and _blocker_text("  ") == ""


def test_the_judge_is_told_to_read_web_wording_against_the_supplied_documents():
    assert "supplied documents are the pages" in JUDGE_SYSTEM
    assert "never write \"none\"" in JUDGE_SYSTEM
    assert "a clarification trigger is never a blocker" in JUDGE_SYSTEM
    assert "required_receipts N" in JUDGE_SYSTEM
