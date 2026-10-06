"""Leases: allowed action classes, budget cap, size policy, class enforcement (scripted, offline).

Sec 5.6 sizes, Sec 6.1 lease fields, Sec 7.2 "action restrictions are enforced".
"""
from dataclasses import replace
from typing import Dict, List

import pytest

from cglc import Runner, TaskContract
from cglc.config import BudgetLimits, CGLCConfig, DEFAULT_CONFIG, LeaseConfig
from cglc.controller import lease_category_for, pick_lease_category
from cglc.leases import (ACTION_CLASSES, ANSWER, INTENT_ACTIONS, READ, SEARCH, VERIFY, Lease,
                         action_permitted, lease_budget_cap)
from cglc.ledger import EvidenceLedger
from cglc.llm import LLMReply, LLMUsage
from cglc.runner import Judgment
from cglc.scoring import ActionFeatures
from cglc.trace import Trace
from cglc.worker.adapter import WorkerAdapter
from cglc.worker.base import DocumentWorker, Observation, WorkerResult
from cglc.worker.extractive import ExtractiveWorker
from cglc.worker.fixed_corpus import FixedCorpusWorker
from cglc.worker.llm_worker import LLMDocumentWorker

COST = {"tool_calls": 1.0, "tokens": 100.0, "wall_clock": 1.0}


class ScriptedWorker(DocumentWorker):
    """Each act() yields one fresh passage (so no stall) and a scripted class/cost."""

    def __init__(self, cost: Dict[str, float] | None = None, classes: List[str] | None = None,
                 finals: tuple = ()):
        self.cost = dict(COST if cost is None else cost)
        self.classes = classes
        self.finals = finals
        self.calls: List[tuple] = []
        self.notes: List[str] = []

    def act(self, intent, target_gaps, allowed_classes, draft):
        n = len(self.calls)
        self.calls.append((intent, list(target_gaps), list(allowed_classes)))
        self.notes.append(self.controller_note)
        cls = (self.classes[min(n, len(self.classes) - 1)] if self.classes else SEARCH)
        obs = [Observation(source_id="d", span_id=f"d#c{n}", text=f"passage {n}", chunk_id=f"d#c{n}")]
        return WorkerResult(observations=obs, draft=f"draft {n}", propose_final=n in self.finals,
                            detail={"action_text": f"query{n} topic{n}", "cost": dict(self.cost),
                                    "action_class": cls})


def feats():
    return {"CONTINUE": ActionFeatures(delta=0.5, G=1.0, d=0.05),
            "VERIFY": ActionFeatures(delta=0.0, V=0.2, d=0.03),
            "REDIRECT": ActionFeatures(delta=0.0, R=0.1, d=0.08)}


class ScriptedJudge:
    """Checkpoint n applies steps[n] (the last one repeats) to the ledger."""

    def __init__(self, steps):
        self.steps = steps
        self.n = 0

    def __call__(self, contract, ledger, draft):
        st = self.steps[min(self.n, len(self.steps) - 1)]
        self.n += 1
        for oid, v in st.get("s", {}).items():
            ledger.s[oid] = v
        for oid, v in st.get("c", {}).items():
            ledger.c[oid] = v
        return Judgment(
            evidence_sufficient=st.get("suff", False), answer_conforms=True,
            weak_or_contested=st.get("contested", False),
            open_gaps=list(st.get("open", [])), action_features=st.get("feats") or feats(),
            rationale=st.get("rationale", "r"))


def contract(n: int = 1, duties: int = 0) -> TaskContract:
    return TaskContract.create("goal", [f"do {i}" for i in range(duties)],
                               [f"claim {i}" for i in range(n)])


def run(c, worker, judge, cfg=DEFAULT_CONFIG, process_check=lambda: (True, []), **kw):
    trace = Trace()
    adapter = WorkerAdapter(worker, trace)
    led = EvidenceLedger([o.obligation_id for o in c.evidence_obligations])
    res = Runner(cfg=cfg, judge=judge, harvest_receipts=False, **kw).run(
        c, adapter, led, trace, process_check=process_check)
    return res, trace, adapter, led


def done(open_=()):
    """Judge steps: not sufficient first, then sufficient."""
    return [{"open": list(open_)}, {"suff": True}]


# --- A. allowed action classes by intent ------------------------------------------------
def test_lease_defaults_allowed_classes_from_intent():
    assert Lease.make("CONTINUE").allowed_action_classes == [SEARCH, READ, ANSWER]
    assert Lease.make("VERIFY").allowed_action_classes == [SEARCH, READ, VERIFY, ANSWER]
    assert Lease.make("REDIRECT").allowed_action_classes == [SEARCH, READ, ANSWER]
    assert set(INTENT_ACTIONS) == {"CONTINUE", "VERIFY", "REDIRECT"}
    # VERIFY keeps SEARCH so a verification lease cannot loop uselessly
    assert SEARCH in INTENT_ACTIONS["VERIFY"] and VERIFY not in INTENT_ACTIONS["CONTINUE"]
    assert Lease.make("SOMETHING_NEW").allowed_action_classes == list(ACTION_CLASSES)
    # an explicit empty list means unrestricted, and None takes the default
    unrestricted = Lease.make("CONTINUE", allowed=[])
    assert unrestricted.allowed_action_classes == [] and unrestricted.permits(VERIFY)
    assert not Lease.make("CONTINUE").permits(VERIFY) and Lease.make("VERIFY").permits(VERIFY)
    assert action_permitted([], "ANYTHING") and action_permitted(None, SEARCH)


def test_runner_issues_leases_with_intent_classes():
    w = ScriptedWorker()
    res, *_ = run(contract(2), w, ScriptedJudge(done(["ev-0"])), process_check=lambda: (True, []))
    assert w.calls[0][2] == [SEARCH, READ, ANSWER]
    assert res.records[0].next_lease["allowed_action_classes"] == [SEARCH, READ, ANSWER]
    w = ScriptedWorker()
    res, *_ = run(contract(2), w, ScriptedJudge([{"contested": True, "open": ["ev-0"]}, {"suff": True}]),
                  policy="rules")
    assert res.records[0].decision == "VERIFY"
    assert res.records[0].next_lease["allowed_action_classes"] == [SEARCH, READ, VERIFY, ANSWER]
    assert VERIFY in w.calls[-1][2]


# --- C. lease budget cap -----------------------------------------------------------------
def test_budget_cap_is_derived_from_remaining_budget_and_category_share():
    rem = {"tool_calls": 20.0, "tokens": 40000.0, "wall_clock": 300.0}
    cap = lease_budget_cap("STANDARD", 3, rem, DEFAULT_CONFIG.lease.budget_share)
    assert cap == {"tool_calls": 3.0, "tokens": 14000.0, "wall_clock": 105.0}
    assert lease_budget_cap("EXTENDED", 5, rem, DEFAULT_CONFIG.lease.budget_share)["tokens"] == 20000.0
    assert lease_budget_cap("SHORT", 1, rem, DEFAULT_CONFIG.lease.budget_share)["tokens"] == 8000.0
    assert lease_budget_cap("SHORT", 1, {}, {}) == {"tool_calls": 1.0}  # undeclared dimension: uncapped


def test_lease_expires_on_action_cap_or_on_any_spent_capped_dimension():
    cap = Lease.make("CONTINUE", "STANDARD", budget_cap={"tool_calls": 3.0, "tokens": 250.0})
    assert not cap.expired and cap.expiry_reason == ""
    assert cap.charge({"tool_calls": 1.0, "tokens": 100.0}) == ""
    assert cap.charge({"tool_calls": 1.0, "tokens": 100.0}) == ""
    assert cap.charge({"tool_calls": 1.0, "tokens": 100.0}) == "action_cap"   # 3 actions
    tokens = Lease.make("CONTINUE", "STANDARD", budget_cap={"tool_calls": 3.0, "tokens": 150.0})
    tokens.charge({"tool_calls": 1.0, "tokens": 100.0})
    assert tokens.charge({"tool_calls": 1.0, "tokens": 100.0}) == "budget_cap" and tokens.expired
    two_calls = Lease.make("CONTINUE", "STANDARD", budget_cap={"tool_calls": 3.0})
    assert two_calls.charge({"tool_calls": 2.0}) == ""
    assert two_calls.charge({"tool_calls": 2.0}) == "budget_cap" and two_calls.actions_used == 2
    both = Lease.make("CONTINUE", "SHORT", budget_cap={"tool_calls": 1.0})
    assert both.charge({"tool_calls": 1.0}) == "action_cap"   # nominal cap wins the label


def test_first_action_is_always_allowed_even_under_a_zero_cap():
    zero = Lease.make("CONTINUE", "STANDARD", budget_cap={"tokens": 0.0, "wall_clock": 0.0})
    assert not zero.expired                      # nothing charged yet: the cap cannot end it
    assert zero.charge({"tokens": 5.0}) == "budget_cap"
    cfg = replace(CGLCConfig(), limits=BudgetLimits(100.0, 1000.0, 1000.0), lease=replace(
        CGLCConfig().lease, budget_share={"SHORT": 0.0, "STANDARD": 0.0, "EXTENDED": 0.0}))
    w = ScriptedWorker()
    res, *_ = run(contract(2), w, ScriptedJudge(done(["ev-0"])), cfg=cfg)
    first = res.records[0]
    assert first.lease["actions_used"] == 1 and first.lease["expiry_reason"] == "budget_cap"
    assert first.lease["budget_cap"]["tokens"] == 0.0


def test_runner_budget_cap_expiry_reports_lease_expiry_with_the_finer_reason_in_the_audit():
    cfg = replace(CGLCConfig(), limits=BudgetLimits(100.0, 1000.0, 1000.0), lease=replace(
        CGLCConfig().lease, budget_share={"SHORT": 0.2, "STANDARD": 0.1, "EXTENDED": 0.1}))
    w = ScriptedWorker(cost={"tool_calls": 1.0, "tokens": 60.0, "wall_clock": 1.0})
    res, trace, *_ = run(contract(2), w, ScriptedJudge(done(["ev-0"])), cfg=cfg)
    r0 = res.records[0]
    # STANDARD cap of 3 actions, but 10% of 1000 tokens = 100: stops after the 2nd action (120 spent)
    assert r0.lease["category"] == "STANDARD" and r0.lease["action_cap"] == 3
    assert r0.lease["actions_used"] == 2 and r0.lease["expiry_reason"] == "budget_cap"
    assert r0.lease["spent"]["tokens"] == 120.0 and r0.lease["budget_cap"]["tokens"] == 100.0
    assert r0.trigger_reasons == ["lease_expiry"]            # the six events are unchanged
    assert len([e for e in trace.events if e.action_class == "WORK"]) >= 2


def test_lease_spent_is_charged_from_the_logged_trace_event():
    w = ScriptedWorker(cost={"tool_calls": 1.0, "tokens": 7.0, "wall_clock": 0.5})
    res, *_ = run(contract(2), w, ScriptedJudge(done(["ev-0"])))
    lease = res.records[0].lease
    assert lease["spent"] == {"tool_calls": 3.0, "tokens": 21.0, "wall_clock": 1.5}
    assert lease["expiry_reason"] == "action_cap"


# --- D. lease size policy (Sec 5.6) -----------------------------------------------------
@pytest.mark.parametrize("decision,n_open,rho,stalled,want", [
    ("VERIFY", 5, 0.0, False, "SHORT"),         # verification
    ("REDIRECT", 5, 0.0, False, "SHORT"),       # redirection
    ("CONTINUE", 1, 0.1, False, "SHORT"),       # near-final: one open item
    ("CONTINUE", 0, 0.1, False, "SHORT"),
    ("CONTINUE", 4, 0.75, False, "SHORT"),      # high budget pressure
    ("CONTINUE", 4, 0.9, False, "SHORT"),
    ("CONTINUE", 3, 0.25, False, "EXTENDED"),   # several independent items, low pressure
    ("CONTINUE", 5, 0.0, False, "EXTENDED"),
    ("CONTINUE", 3, 0.25, True, "STANDARD"),    # a stalled direction never gets EXTENDED
    ("CONTINUE", 3, 0.26, False, "STANDARD"),   # pressure above extended_max_pressure
    ("CONTINUE", 2, 0.3, False, "STANDARD"),    # default
    ("CONTINUE", 2, 0.74, False, "STANDARD"),
])
def test_pick_lease_category_policy(decision, n_open, rho, stalled, want):
    gaps = [f"ev-{i}" for i in range(n_open)]
    assert pick_lease_category(decision, gaps, n_open, rho, stalled, DEFAULT_CONFIG) == want
    assert pick_lease_category(decision, n_open, None, rho, stalled, DEFAULT_CONFIG) == want  # counts work too


def test_pick_lease_category_thresholds_are_configuration_and_early_stage_skips_near_final():
    cfg = replace(DEFAULT_CONFIG, lease=replace(DEFAULT_CONFIG.lease, extended_min_gaps=2,
                                                short_max_gaps=0, short_min_pressure=0.5))
    assert pick_lease_category("CONTINUE", [], 2, 0.1, False, cfg) == "EXTENDED"
    assert pick_lease_category("CONTINUE", [], 1, 0.1, False, cfg) == "STANDARD"
    assert pick_lease_category("CONTINUE", [], 2, 0.5, False, cfg) == "SHORT"
    # first lease: one open obligation is the whole task, not "near-final"
    assert pick_lease_category("CONTINUE", ["ev-0"], 1, 0.0, False, DEFAULT_CONFIG) == "SHORT"
    assert pick_lease_category("CONTINUE", ["ev-0"], 1, 0.0, False, DEFAULT_CONFIG,
                               early_stage=True) == "STANDARD"
    assert pick_lease_category("CONTINUE", ["ev-0"], 1, 0.9, False, DEFAULT_CONFIG,
                               early_stage=True) == "SHORT"       # pressure still wins


def test_lease_category_for_is_kept_for_callers():
    assert lease_category_for("VERIFY") == "SHORT" and lease_category_for("REDIRECT") == "SHORT"
    assert lease_category_for("CONTINUE") == "STANDARD"
    assert lease_category_for("CONTINUE", near_final=True) == "SHORT"
    assert lease_category_for("CONTINUE", early_stage=True) == "EXTENDED"


def test_first_lease_is_extended_for_three_obligations_and_standard_for_fewer():
    w = ScriptedWorker()
    res, *_ = run(contract(3), w, ScriptedJudge(done(["ev-0"])))
    r0 = res.records[0]
    assert r0.lease["category"] == "EXTENDED" and r0.lease["action_cap"] == 5
    assert r0.lease["actions_used"] == 5 and len(w.calls) >= 5
    assert "max_silence" in r0.trigger_reasons           # the silence backstop stays active
    for n in (1, 2):
        res, *_ = run(contract(n), ScriptedWorker(), ScriptedJudge(done(["ev-0"])))
        assert res.records[0].lease["category"] == "STANDARD"


def test_later_leases_follow_the_policy_near_final_pressure_and_default():
    # 3 open items, early: EXTENDED; one left: SHORT (near-final)
    res, *_ = run(contract(3), ScriptedWorker(), ScriptedJudge(done(["ev-2"])))
    assert [res.records[0].lease["category"], res.records[0].next_lease["category"]] == ["EXTENDED", "SHORT"]
    # two left: STANDARD
    res, *_ = run(contract(3), ScriptedWorker(), ScriptedJudge(done(["ev-1", "ev-2"])))
    assert res.records[0].next_lease["category"] == "STANDARD"
    # high pressure: 3 of 4 tool calls used -> rho 0.75 -> SHORT despite two open items
    cfg = replace(CGLCConfig(), limits=BudgetLimits(4.0, 100000.0, 1000.0))
    res, *_ = run(contract(2), ScriptedWorker(), ScriptedJudge(done(["ev-0", "ev-1"])), cfg=cfg)
    assert res.records[0].next_lease["category"] == "SHORT"
    # unmet process duties count as open items
    res, *_ = run(contract(2, duties=2), ScriptedWorker(), ScriptedJudge(done(["ev-0"])),
                  process_check=lambda: (False, ["proc-0", "proc-1"]))
    assert res.records[0].next_lease["target_gap_ids"][:2] == ["duty:proc-0", "duty:proc-1"]
    assert res.records[0].next_lease["category"] == "EXTENDED"     # 3 open items, low pressure


def test_fixed_lease_still_forces_every_lease_and_ignores_the_policy():
    w = ScriptedWorker()
    res, trace, *_ = run(contract(4), w, ScriptedJudge([{"open": ["ev-0", "ev-1", "ev-2", "ev-3"]}] * 4 + [{"suff": True}]),
                         policy="rules", fixed_lease=2, max_checkpoints=6)
    leases = [r.lease for r in res.records]
    assert leases and all(l["action_cap"] == 2 and l["category"] == "FIXED" for l in leases)
    assert all(l["budget_cap"] == {} and l["actions_used"] == 2 for l in leases[:-1])
    assert res.records[0].next_lease["action_cap"] == 2
    # always-on judge: one action per lease, a checkpoint after every action
    w = ScriptedWorker()
    res, trace, *_ = run(contract(2), w, ScriptedJudge([{"open": ["ev-0"]}] * 3 + [{"suff": True}]),
                         fixed_lease=1, max_checkpoints=6)
    work = [e for e in trace.events if e.action_class == "WORK"]
    assert len(res.records) == len(work)


def test_contract_supplied_lease_action_caps_still_define_the_sizes():
    cfg = replace(CGLCConfig(), lease=replace(CGLCConfig().lease, SHORT=2, STANDARD=2, EXTENDED=4))
    res, *_ = run(contract(3), ScriptedWorker(), ScriptedJudge(done(["ev-0"])), cfg=cfg)
    r0 = res.records[0]
    assert (r0.lease["category"], r0.lease["action_cap"], r0.lease["actions_used"]) == ("EXTENDED", 4, 4)
    assert r0.lease["budget_cap"]["tool_calls"] == 4.0
    assert r0.next_lease["category"] == "SHORT" and r0.next_lease["action_cap"] == 2


def test_lease_config_shows_up_in_the_logged_config():
    d = CGLCConfig().to_dict()["lease"]
    assert d["budget_share"] == {"SHORT": 0.20, "STANDARD": 0.35, "EXTENDED": 0.50}
    assert (d["short_max_gaps"], d["short_min_pressure"]) == (1, 0.75)
    assert (d["extended_min_gaps"], d["extended_max_pressure"]) == (3, 0.25)
    assert (d["SHORT"], d["STANDARD"], d["EXTENDED"], d["L_max"]) == (1, 3, 5, 5)
    assert CGLCConfig().to_dict()["budget_control"]["use_eff_in_delta"] is True
    assert LeaseConfig().budget_share["EXTENDED"] > LeaseConfig().budget_share["SHORT"]


# --- B. action-class enforcement (worker boundary + adapter verification) ----------------
def test_adapter_records_and_counts_a_reported_class_the_lease_does_not_allow():
    trace = Trace()
    adapter = WorkerAdapter(ScriptedWorker(classes=[VERIFY, SEARCH]), trace)
    adapter.act("CONTINUE", ["ev-0"], [SEARCH, READ, ANSWER], "")      # reports VERIFY: not allowed
    adapter.act("CONTINUE", ["ev-0"], [SEARCH, READ, ANSWER], "")      # reports SEARCH: allowed
    bad, ok = trace.events
    assert bad.status == "lease_violation" and bad.detail["action_class"] == VERIFY
    assert bad.detail["allowed_classes"] == [SEARCH, READ, ANSWER]
    assert ok.status == "ok" and ok.detail["action_class"] == SEARCH
    assert adapter.violations == 1
    assert adapter.violation_log == [{"event_id": 0, "intent": "CONTINUE", "action_class": VERIFY,
                                      "allowed_classes": [SEARCH, READ, ANSWER]}]


def test_empty_allowed_list_is_unrestricted_and_unreported_class_is_not_a_violation():
    trace = Trace()
    adapter = WorkerAdapter(ScriptedWorker(classes=[VERIFY]), trace)
    adapter.act("CONTINUE", [], [], "")
    assert trace.events[0].status == "ok" and adapter.violations == 0

    class Silent(DocumentWorker):
        def act(self, intent, target_gaps, allowed_classes, draft):
            return WorkerResult(observations=[], draft=draft)

    t2 = Trace()
    a2 = WorkerAdapter(Silent(), t2)
    a2.act("CONTINUE", [], [SEARCH], "")
    assert t2.events[0].status == "ok" and t2.events[0].detail["action_class"] == "" and a2.violations == 0


def test_a_violating_final_proposal_is_still_a_final_proposal_for_the_runner():
    w = ScriptedWorker(classes=[VERIFY], finals=(0,))
    res, trace, adapter, _ = run(contract(2), w, ScriptedJudge(done(["ev-0"])))
    assert res.records[0].trigger_reasons == ["finalize_request"]
    assert trace.events[0].status == "lease_violation" and trace.events[0].detail["propose_final"] is True


def test_runner_books_violations_on_the_lease_and_the_audit_note():
    w = ScriptedWorker(classes=[VERIFY, SEARCH, VERIFY] + [SEARCH] * 10)
    res, trace, adapter, _ = run(contract(2), w, ScriptedJudge(done(["ev-0"])))
    r0 = res.records[0]
    assert r0.lease["violations"] == 2 == adapter.violations
    assert "lease violations: 2" in r0.note
    assert res.records[0].lease["actions_used"] == 3


def test_a_class_the_lease_allows_is_not_a_violation():
    # a VERIFY lease permits VERIFY
    w = ScriptedWorker(classes=[SEARCH] * 3 + [VERIFY] * 5)
    res, trace, adapter, _ = run(contract(2), w, ScriptedJudge([{"contested": True, "open": ["ev-0"]},
                                                                {"suff": True}]),
                                 policy="rules")
    assert res.records[0].decision == "VERIFY"
    assert adapter.violations == 0 and all(e.status != "lease_violation" for e in trace.events)


# --- B. LLM worker (whole-document mode) -------------------------------------------------
DOCS = {"A": "Architecture A uses an always-on semantic coverage layer. It is expensive."}


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.users: List[str] = []

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        self.users.append(user)
        return LLMReply({"query": "q", "citations": [], "draft": "d", "propose_final": False,
                         "contradiction": False, "blocker": ""},
                        LLMUsage(input_tokens=10, output_tokens=5), 0.0)


def test_llm_worker_stores_classes_reports_class_and_tells_the_model_what_is_permitted():
    llm = FakeLLM()
    w = LLMDocumentWorker(DOCS, contract(1), llm)
    res = w.act("CONTINUE", ["ev-0"], [SEARCH, READ, ANSWER], "")
    assert w.allowed_classes == [SEARCH, READ, ANSWER] and res.detail["action_class"] == READ
    assert "LEASE PERMITS ACTION CLASSES" in llm.users[0] and "Do not spend this step on: VERIFY" in llm.users[0]
    res = w.act("VERIFY", ["ev-0"], [SEARCH, READ, VERIFY, ANSWER], "")
    assert res.detail["action_class"] == VERIFY
    assert "Do not spend this step on" not in llm.users[1] and "VERIFY (re-check" in llm.users[1]
    # VERIFY intent but VERIFY not permitted -> the worker reports what it can do: READ
    assert w.act("VERIFY", ["ev-0"], [SEARCH, READ, ANSWER], "").detail["action_class"] == READ
    # unrestricted: no restriction line, class still reported
    res = w.act("VERIFY", ["ev-0"], [], "")
    assert w.allowed_classes == [] and res.detail["action_class"] == VERIFY
    assert "LEASE PERMITS" not in llm.users[3]
    # the permission line never impersonates the controller note
    assert "CONTROLLER NOTE" not in llm.users[0]


def test_llm_worker_class_reaches_the_trace_through_the_adapter():
    trace = Trace()
    adapter = WorkerAdapter(LLMDocumentWorker(DOCS, contract(1), FakeLLM()), trace)
    adapter.act("CONTINUE", ["ev-0"], [SEARCH, READ, ANSWER], "")
    adapter.act("VERIFY", ["ev-0"], [SEARCH, READ, VERIFY, ANSWER], "")
    assert [e.detail["action_class"] for e in trace.events] == [READ, VERIFY]
    assert adapter.violations == 0


def test_retrieval_workers_report_search_and_one_action_is_one_tool_call():
    docs = {"a": "alpha beta gamma delta", "b": "beta gamma delta epsilon", "c": "gamma delta zeta"}
    fc = FixedCorpusWorker(docs, top_k=3)
    res = fc.act("CONTINUE", ["alpha"], [SEARCH, READ, ANSWER], "")
    assert res.detail["action_class"] == SEARCH and fc.allowed_classes == [SEARCH, READ, ANSWER]
    assert sum(o.cost["tool_calls"] for o in res.observations) == pytest.approx(1.0)
    ex = ExtractiveWorker({"a": "alpha beta gamma delta. " * 5, "b": "epsilon zeta eta theta. " * 5},
                          query="alpha beta", top_k=2)
    res = ex.act("CONTINUE", [], [SEARCH], "")
    assert res.detail["action_class"] == SEARCH and ex.allowed_classes == [SEARCH]
    assert sum(o.cost["tool_calls"] for o in res.observations) == pytest.approx(1.0)
    # a reported class outside a restricted lease is caught by the adapter, not trusted
    trace = Trace()
    WorkerAdapter(fc, trace).act("CONTINUE", [], [READ], "")
    assert trace.events[0].status == "lease_violation"


# --- the CONTROLLER NOTE flow is untouched --------------------------------------------------
def test_judge_rationale_still_becomes_the_expected_progress_test_and_the_controller_note():
    w = ScriptedWorker()
    steps = [{"open": ["ev-0"], "rationale": "cited span only mentions Santa Fe"}, {"suff": True}]
    res, *_ = run(contract(2), w, ScriptedJudge(steps))
    assert w.notes[0] == ""                                   # first lease: nothing to say yet
    assert w.notes[3] == "cited span only mentions Santa Fe"   # the next lease carries the judge's reason
    nxt = res.records[0].next_lease
    assert nxt["expected_progress_test"] == "cited span only mentions Santa Fe"
    # a fail-closed judge's text is not forwarded
    w = ScriptedWorker()
    res, *_ = run(contract(2), w, ScriptedJudge([{"open": ["ev-0"], "rationale": "judge fail-closed: down"},
                                                 {"suff": True}]))
    assert set(w.notes) == {""}
    assert res.records[0].next_lease["expected_progress_test"] == ""
