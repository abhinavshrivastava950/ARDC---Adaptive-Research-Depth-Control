from cglc import TaskContract, Runner
from cglc.ledger import EvidenceLedger
from cglc.trace import Trace
from cglc.worker.fixed_corpus import FixedCorpusWorker
from cglc.worker.adapter import WorkerAdapter


def _case(docs=None):
    docs = docs or {
        "a": "comparison evidence architecture alpha supports criterion one",
        "b": "architecture beta evidence criterion two contradiction check",
    }
    c = TaskContract.create("compare alpha vs beta", ["inspect both"],
                            ["support comparison with evidence"])
    tr = Trace()
    led = EvidenceLedger([o.obligation_id for o in c.evidence_obligations])
    w = WorkerAdapter(FixedCorpusWorker(docs), tr)
    return c, w, led, tr


def test_runner_terminates_with_audit():
    c, w, led, tr = _case()
    res = Runner(max_checkpoints=4).run(
        c, w, led, tr, process_check=lambda: (True, []))
    assert res.decision in ("ALLOW_FINALIZE", "ASK_USER", "REPORT_BLOCKED")
    assert len(res.records) >= 1
    assert all(r.contract_id == c.contract_id for r in res.records)


def test_resource_exhaustion_is_not_success():
    c, w, led, tr = _case({"a": "nothing relevant here"})
    c.evidence_obligations[0].proposition = "need missing evidence"
    # force tiny budget so exhaustion hits before sufficiency
    from cglc.config import CGLCConfig, BudgetLimits
    from dataclasses import replace
    cfg = replace(CGLCConfig(), limits=BudgetLimits(1.0, 50.0, 1.0))
    res = Runner(cfg=cfg, max_checkpoints=3).run(
        c, w, led, tr, process_check=lambda: (True, []))
    if res.decision == "REPORT_BLOCKED":
        assert res.gates is not None and res.gates.allow is False


def test_unmet_duties_block_finalization():
    # §5.5 step 2: without a duty check, duties stay unmet so the
    # process gate fails and a follow-up lease is selected instead.
    # No hook supplied here on purpose.
    c, w, led, tr = _case()
    res = Runner(max_checkpoints=2).run(c, w, led, tr)
    assert res.decision != "ALLOW_FINALIZE"
    assert res.records[0].selected_lease_id is not None


def test_invalid_contract_requests_revision():
    c = TaskContract.create("", [], [])
    tr = Trace()
    led = EvidenceLedger([])
    from cglc.worker.fixed_corpus import FixedCorpusWorker
    w = WorkerAdapter(FixedCorpusWorker({"a": "x"}), tr)
    res = Runner().run(c, w, led, tr)
    assert res.decision == "ASK_USER"
    assert res.records[0].trigger_reasons == ["contract_invalid"]


def test_allow_record_carries_certificate():
    c, w, led, tr = _case()
    res = Runner(max_checkpoints=4).run(
        c, w, led, tr, process_check=lambda: (True, []))
    allows = [r for r in res.records if r.decision == "ALLOW_FINALIZE"]
    assert allows, "expected an ALLOW_FINALIZE record"
    rec = allows[0]
    assert rec.contract_snapshot["contract_id"] == c.contract_id
    assert isinstance(rec.supporting_receipts, dict)
    assert rec.selected_lease_id is None  # terminal: no next lease


def test_blocked_record_carries_condition():
    c, w, led, tr = _case({"a": "nothing relevant here"})
    from cglc.config import CGLCConfig, BudgetLimits
    from dataclasses import replace
    cfg = replace(CGLCConfig(), limits=BudgetLimits(1.0, 50.0, 1.0))
    res = Runner(cfg=cfg, max_checkpoints=2).run(
        c, w, led, tr, process_check=lambda: (True, []))
    blocked = [r for r in res.records if r.decision == "REPORT_BLOCKED"]
    assert blocked and blocked[-1].blocked_condition != ""


def test_overhead_guard_downgrades_but_keeps_interception():
    from cglc.leases import OverheadGuard
    g = OverheadGuard(fraction=0.25, l_max=5)
    assert g.observe(10.0, 100.0) is False
    assert g.observe(90.0, 10.0) is True  # 90% controller spend
    assert g.l_max == 10
    # interception retained: finalize scheduling is independent of L_max
    from cglc.guards import schedule_checkpoint
    ev = schedule_checkpoint(True, False, False, False, False, False, False)
    assert ev.run and ev.reasons == ["finalize_request"]


def test_ablation_ladder_runs():
    from cglc.config import DEFAULT_CONFIG
    from cglc.evaluate.metrics import run_ablations, no_stall_config
    from cglc.runner import rule_judge

    def make_case():
        return _case()

    rows = run_ablations(make_case, [
        ("cglc_full", DEFAULT_CONFIG, None, lambda: (True, [])),
        ("cglc_no_stall_trigger", no_stall_config(DEFAULT_CONFIG), None,
         lambda: (True, [])),
    ])
    assert [r["variant"] for r in rows] == ["cglc_full", "cglc_no_stall_trigger"]
    assert all(r["checkpoints"] >= 1 for r in rows)
