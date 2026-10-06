"""Eff_support / Eff_resolve, the lease progress test and the audit fields (scripted, offline).

Sec 14.3: the two quantities are reported separately and weighted by w_i.
Sec 5.6 / 7.5: the lease that ran, its progress-test result and the next lease are audited.
Q&A Q3/Q4: Eff feeds the progress estimate Delta (flagged, ranking only).
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_leases import ScriptedJudge, ScriptedWorker, feats, run  # noqa: E402

from cglc import TaskContract  # noqa: E402
from cglc.config import CGLCConfig, DEFAULT_CONFIG  # noqa: E402
from cglc.evaluate.metrics import summarize  # noqa: E402
from cglc.leases import PROGRESS_RULE, Lease  # noqa: E402
from cglc.runner import Judgment, Runner  # noqa: E402
from cglc.scoring import ActionFeatures  # noqa: E402


def weighted_contract() -> TaskContract:
    return TaskContract.from_dict({"goal": "g", "evidence_obligations": [
        {"obligation_id": "a", "proposition": "claim a", "weight": 2.0},
        {"obligation_id": "b", "proposition": "claim b", "weight": 0.5},
        {"obligation_id": "c", "proposition": "claim c", "weight": 1.0}]})


# --- Eff_support / Eff_resolve ---------------------------------------------------------
def test_eff_is_weighted_reported_separately_and_present_on_terminal_and_nonterminal_records():
    steps = [
        {"s": {"a": 1.0, "b": 0.5}, "c": {"c": 1}, "open": ["b", "c"]},   # a UNSEEN->SUPPORTED, b->PARTIAL; c contested
        {"s": {"b": 1.0}, "c": {"c": 0}, "open": ["b"]},                  # b->SUPPORTED; c's contradiction resolved
        {"s": {"b": 1.0, "c": 1.0}, "suff": True},                        # c->SUPPORTED: final checkpoint
    ]
    res, *_ = run(weighted_contract(), ScriptedWorker(), ScriptedJudge(steps))
    r1, r2, r3 = res.records
    assert (r1.eff_support, r1.eff_resolve) == (2.0 * 1.0 + 0.5 * 0.5, 0.0)    # 2.25: weights applied
    assert (r2.eff_support, r2.eff_resolve) == (0.5 * 0.5, 1.0 * 1)            # support and resolve differ
    assert r3.decision == "ALLOW_FINALIZE" and (r3.eff_support, r3.eff_resolve) == (1.0, 0.0)
    assert r1.next_lease is not None and r2.next_lease is not None and r3.next_lease is None
    # reported as two fields, never merged
    assert "eff_support" in r1.to_dict() and "eff_resolve" in r1.to_dict()
    assert json.loads(r2.to_json())["eff_resolve"] == 1.0


def test_eff_is_measured_against_the_previous_checkpoint_not_the_start():
    steps = [{"s": {"a": 1.0}, "open": ["b", "c"]}, {"open": ["b", "c"]}, {"suff": True, "s": {"b": 1.0, "c": 1.0}}]
    res, *_ = run(weighted_contract(), ScriptedWorker(), ScriptedJudge(steps))
    assert [r.eff_support for r in res.records] == [2.0, 0.0, 1.5]   # a stays SUPPORTED: no new progress


def test_default_weights_are_one():
    c = TaskContract.create("g", [], ["x", "y"])
    res, *_ = run(c, ScriptedWorker(), ScriptedJudge([{"s": {"ev-0": 1.0, "ev-1": 0.5}, "suff": True}]))
    assert res.records[0].eff_support == 1.5


# --- lease progress test ----------------------------------------------------------------
def test_progress_test_met_when_a_target_obligation_moves_up():
    steps = [{"s": {"a": 1.0}, "open": ["b", "c"]}, {"suff": True}]
    res, *_ = run(weighted_contract(), ScriptedWorker(), ScriptedJudge(steps))
    p = res.records[0].lease_progress
    assert p["met"] is True and p["progressed"] == ["a"] and p["rule"] == PROGRESS_RULE
    assert p["targets"] == ["a", "b", "c"]                       # the first lease targets every obligation
    assert (p["eff_support"], p["eff_resolve"]) == (2.0, 0.0)
    assert p["duties_cleared"] == []


def test_progress_test_not_met_when_nothing_targeted_moved():
    steps = [{"open": ["b"]}, {"open": ["b"]}, {"suff": True}]
    res, *_ = run(weighted_contract(), ScriptedWorker(), ScriptedJudge(steps))
    first, second = res.records[0].lease_progress, res.records[1].lease_progress
    assert first["met"] is False and first["progressed"] == [] and (first["eff_support"], first["eff_resolve"]) == (0, 0)
    assert second["met"] is False and second["targets"] == ["b"]


def test_progress_test_is_restricted_to_the_lease_targets():
    steps = [
        {"s": {"a": 1.0}, "c": {"c": 1}, "open": ["b"]},     # next lease targets only b
        {"c": {"c": 0}, "open": ["b"]},                      # c resolved: global progress, not b's
        {"suff": True},
    ]
    res, *_ = run(weighted_contract(), ScriptedWorker(), ScriptedJudge(steps))
    r2 = res.records[1]
    assert r2.eff_resolve == 1.0                              # the checkpoint saw the resolution
    assert r2.lease_progress["met"] is False                  # but the lease targeted only b
    assert r2.lease_progress["eff_resolve"] == 0.0 and r2.lease_progress["progressed"] == []


def test_progress_test_counts_a_resolved_contradiction_on_a_target():
    steps = [{"c": {"b": 1}, "open": ["b"]}, {"c": {"b": 0}, "open": ["b"]}, {"suff": True}]
    res, *_ = run(weighted_contract(), ScriptedWorker(), ScriptedJudge(steps))
    p = res.records[1].lease_progress
    assert p["met"] is True and p["progressed"] == ["b"] and p["eff_resolve"] == 0.5 and p["eff_support"] == 0.0


def test_progress_test_met_when_a_targeted_duty_is_cleared():
    c = TaskContract.create("g", ["read all"], ["x", "y"])
    unmet = iter([["proc-0"], [], []])
    res, *_ = run(c, ScriptedWorker(), ScriptedJudge([{"open": ["ev-0", "ev-1"]}] * 2 + [{"suff": True}]),
                  process_check=lambda: (not (u := next(unmet)), u))
    assert res.records[0].next_lease["target_gap_ids"][0] == "duty:proc-0"
    p = res.records[1].lease_progress
    assert p["met"] is True and p["duties_cleared"] == ["proc-0"] and p["progressed"] == []


def test_lease_evaluate_progress_directly():
    lease = Lease.make("CONTINUE", target_gaps=["a", "duty:d1", "ghost"])
    out = lease.evaluate_progress({"a": 0.0}, {"a": 0}, {"a": 0.5}, {"a": 0}, {"a": 4.0}, unmet_duties=["d1"])
    assert out["met"] is True and out["eff_support"] == 2.0 and out["duties_cleared"] == []
    out = lease.evaluate_progress({"a": 0.5}, {"a": 0}, {"a": 0.5}, {"a": 0}, None, unmet_duties=["d1"])
    assert out["met"] is False                                # nothing moved, duty still unmet
    assert lease.evaluate_progress({"a": 0.5}, {"a": 0}, {"a": 0.0}, {"a": 0}, None, ["d1"])["met"] is False  # a regression is not progress


# --- contract hash ----------------------------------------------------------------------
def test_contract_hash_is_present_stable_and_content_based():
    c = TaskContract.create("g", [], ["x", "y"])
    res, *_ = run(c, ScriptedWorker(), ScriptedJudge([{"open": ["ev-0"]}, {"suff": True}]))
    hashes = {r.contract_hash for r in res.records}
    assert hashes == {c.content_hash()} and next(iter(hashes)).startswith("sha256:")
    c2 = TaskContract.create("g", [], ["x", "y"])             # different random contract_id, same content
    res2, *_ = run(c2, ScriptedWorker(), ScriptedJudge([{"suff": True}]))
    assert res2.records[0].contract_hash == c.content_hash()
    c3 = TaskContract.create("another goal", [], ["x", "y"])
    assert run(c3, ScriptedWorker(), ScriptedJudge([{"suff": True}]))[0].records[0].contract_hash != c.content_hash()


def test_invalid_contract_record_also_carries_the_hash_and_no_lease():
    bad = TaskContract.create("", [], [])
    res, *_ = run(bad, ScriptedWorker(), ScriptedJudge([{}]))
    r = res.records[0]
    assert r.trigger_reasons == ["contract_invalid"] and r.contract_hash == bad.content_hash()
    assert r.lease is None and r.lease_progress is None and r.next_lease is None
    assert (r.eff_support, r.eff_resolve) == (0.0, 0.0)


# --- audit record shape -----------------------------------------------------------------
def test_records_carry_the_lease_that_ran_and_the_lease_it_issued():
    res, *_ = run(TaskContract.create("g", [], ["x", "y", "z"]), ScriptedWorker(),
                  ScriptedJudge([{"open": ["ev-1"]}, {"suff": True}]))
    r0, r1 = res.records
    ran, nxt = r0.lease, r0.next_lease
    for k in ("lease_id", "intent", "category", "target_gap_ids", "allowed_action_classes", "action_cap",
              "actions_used", "budget_cap", "spent", "expiry_reason", "violations", "expected_progress_test"):
        assert k in ran and k in nxt
    assert ran["lease_id"] == r0.lease_id and nxt["lease_id"] == r0.selected_lease_id
    assert nxt["actions_used"] == 0 and nxt["spent"] == {} and nxt["expiry_reason"] == ""
    assert r1.lease["lease_id"] == nxt["lease_id"] and r1.next_lease is None   # terminal
    assert r1.lease["actions_used"] > 0 and r1.lease_progress is not None
    json.dumps(r0.to_dict())                                    # the whole record is JSON-serialisable


def test_summarize_reports_eff_separately():
    steps = [{"s": {"a": 1.0}, "c": {"c": 1}, "open": ["b", "c"]}, {"c": {"c": 0}, "open": ["b"]}, {"suff": True}]
    res, *_ = run(weighted_contract(), ScriptedWorker(), ScriptedJudge(steps))
    s = summarize(res.records)
    assert s["eff_support_total"] == 2.0 and s["eff_resolve_total"] == 1.0
    assert s["lease_progress_tested"] == 3 and s["lease_progress_met"] >= 1 and s["lease_violations"] == 0


# --- Eff feeds Delta (flagged ranking input) --------------------------------------------
def verify_friendly_feats():
    """CONTINUE r ~ 12.8 beats VERIFY r ~ 6.0, but with Delta capped at LOW (r ~ 2.8) VERIFY wins."""
    return {"CONTINUE": ActionFeatures(delta=0.5, G=1.0, d=0.05),
            "VERIFY": ActionFeatures(delta=0.0, V=1.0, d=0.03),
            "REDIRECT": ActionFeatures(delta=0.0, d=0.08)}


def eff_cfg(on: bool) -> CGLCConfig:
    return replace(DEFAULT_CONFIG, budget_control=replace(DEFAULT_CONFIG.budget_control, use_eff_in_delta=on))


def test_zero_eff_caps_continue_delta_for_the_ranking_when_the_flag_is_on():
    f = verify_friendly_feats()
    res, *_ = run(TaskContract.create("g", [], ["x", "y"]), ScriptedWorker(),
                  ScriptedJudge([{"open": ["ev-0", "ev-1"], "feats": f}, {"suff": True}]), cfg=eff_cfg(True))
    r0 = res.records[0]
    assert (r0.eff_support, r0.eff_resolve) == (0.0, 0.0)
    assert r0.decision == "VERIFY" and "eff-adjust" in r0.note and "0.5 -> 0" in r0.note
    assert f["CONTINUE"].delta == 0.5                          # the judge's own estimate is not mutated


def test_flag_off_leaves_the_ranking_untouched():
    res, *_ = run(TaskContract.create("g", [], ["x", "y"]), ScriptedWorker(),
                  ScriptedJudge([{"open": ["ev-0", "ev-1"], "feats": verify_friendly_feats()}, {"suff": True}]),
                  cfg=eff_cfg(False))
    assert res.records[0].decision == "CONTINUE" and "eff-adjust" not in res.records[0].note


def test_progress_prevents_the_cap():
    res, *_ = run(TaskContract.create("g", [], ["x", "y"]), ScriptedWorker(),
                  ScriptedJudge([{"s": {"ev-0": 0.5}, "open": ["ev-0", "ev-1"], "feats": verify_friendly_feats()},
                                 {"suff": True}]), cfg=eff_cfg(True))
    assert res.records[0].eff_support == 0.5
    assert res.records[0].decision == "CONTINUE" and "eff-adjust" not in res.records[0].note


def test_no_cap_at_a_finalize_request_only_checkpoint_or_after_a_failed_judge():
    two = TaskContract.create("g", [], ["x", "y"])
    res, *_ = run(two, ScriptedWorker(finals=(0,)),
                  ScriptedJudge([{"open": ["ev-0", "ev-1"], "feats": verify_friendly_feats()}, {"suff": True}]),
                  cfg=eff_cfg(True))
    assert res.records[0].trigger_reasons == ["finalize_request"]
    assert res.records[0].decision == "CONTINUE" and "eff-adjust" not in res.records[0].note
    res, *_ = run(two, ScriptedWorker(),
                  ScriptedJudge([{"open": ["ev-0", "ev-1"], "feats": verify_friendly_feats(),
                                  "rationale": "judge fail-closed: rate limited"}, {"suff": True}]),
                  cfg=eff_cfg(True))
    assert res.records[0].decision == "CONTINUE" and "eff-adjust" not in res.records[0].note


def test_eff_never_authorizes_or_blocks_finalization():
    two = TaskContract.create("g", [], ["x", "y"])
    # zero Eff, but every gate passes: finalization is still allowed
    res, *_ = run(two, ScriptedWorker(), ScriptedJudge([{"suff": True}]), cfg=eff_cfg(True))
    assert res.decision == "ALLOW_FINALIZE" and res.records[0].eff_support == 0.0
    # large Eff, gates fail: no finalization
    res, *_ = run(two, ScriptedWorker(), ScriptedJudge([{"s": {"ev-0": 1.0, "ev-1": 1.0}, "open": ["ev-0"]}] * 3),
                  cfg=eff_cfg(True), max_checkpoints=2)
    assert res.decision != "ALLOW_FINALIZE" and res.records[0].eff_support == 2.0


def test_the_cap_is_ranked_policy_only_and_only_for_continue():
    # rules policy has no value ranking: nothing to adjust
    res, *_ = run(TaskContract.create("g", [], ["x", "y"]), ScriptedWorker(),
                  ScriptedJudge([{"open": ["ev-0", "ev-1"], "feats": verify_friendly_feats()}, {"suff": True}]),
                  cfg=eff_cfg(True), policy="rules")
    assert res.records[0].decision == "CONTINUE" and "eff-adjust" not in res.records[0].note
    # a VERIFY lease that moves nothing does not penalise VERIFY/REDIRECT: only CONTINUE's delta is capped
    rj = {"CONTINUE": ActionFeatures(delta=0.0, d=0.05),
          "VERIFY": ActionFeatures(delta=0.5, V=1.0, d=0.03),
          "REDIRECT": ActionFeatures(delta=0.5, d=0.08)}
    res, *_ = run(TaskContract.create("g", [], ["x", "y"]), ScriptedWorker(),
                  ScriptedJudge([{"open": ["ev-0"], "feats": rj}, {"open": ["ev-0"], "feats": rj}, {"suff": True}]),
                  cfg=eff_cfg(True), max_checkpoints=4)
    assert [r.decision for r in res.records[:2]] == ["VERIFY", "VERIFY"]
    assert "eff-adjust" not in res.records[0].note


def test_existing_judge_features_object_flow_is_unchanged_for_rule_judge_runs():
    # smoke: the default rule judge still drives a run to a terminal decision with the new fields
    from cglc.ledger import EvidenceLedger
    from cglc.trace import Trace
    from cglc.worker.adapter import WorkerAdapter
    from cglc.worker.fixed_corpus import FixedCorpusWorker
    c = TaskContract.create("compare", ["inspect"], ["architecture alpha evidence"])
    tr = Trace()
    led = EvidenceLedger([o.obligation_id for o in c.evidence_obligations])
    w = WorkerAdapter(FixedCorpusWorker({"a": "architecture alpha evidence supports it",
                                         "b": "architecture beta evidence"}), tr)
    res = Runner(max_checkpoints=4).run(c, w, led, tr, process_check=lambda: (True, []))
    assert res.decision == "ALLOW_FINALIZE"
    last = res.records[-1]
    assert last.next_lease is None and last.lease["actions_used"] >= 1 and last.contract_hash == c.content_hash()
    assert last.eff_support > 0 or any(r.eff_support > 0 for r in res.records)
    assert isinstance(Judgment().action_features, dict)
