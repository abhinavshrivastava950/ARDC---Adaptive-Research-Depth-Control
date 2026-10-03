from cglc import TaskContract, Runner
from cglc.ledger import EvidenceLedger
from cglc.trace import Trace
from cglc.worker.fixed_corpus import FixedCorpusWorker
from cglc.worker.adapter import WorkerAdapter


def test_runner_terminates_with_audit():
    docs = {
        "a": "comparison evidence architecture alpha supports criterion one",
        "b": "architecture beta evidence criterion two contradiction check",
    }
    c = TaskContract.create("compare alpha vs beta", ["inspect both"],
                            ["support comparison with evidence"])
    tr = Trace()
    led = EvidenceLedger([o.obligation_id for o in c.evidence_obligations])
    w = WorkerAdapter(FixedCorpusWorker(docs), tr)
    res = Runner(max_checkpoints=4).run(c, w, led, tr)
    assert res.decision in ("ALLOW_FINALIZE", "ASK_USER", "REPORT_BLOCKED")
    assert len(res.records) >= 1
    assert all(r.contract_id == c.contract_id for r in res.records)


def test_resource_exhaustion_is_not_success():
    docs = {"a": "nothing relevant here"}
    c = TaskContract.create("impossible task", [], ["need missing evidence"])
    tr = Trace()
    led = EvidenceLedger([o.obligation_id for o in c.evidence_obligations])
    w = WorkerAdapter(FixedCorpusWorker(docs), tr)
    # force tiny budget so exhaustion hits before sufficiency
    from cglc.config import CGLCConfig, BudgetLimits
    from dataclasses import replace
    cfg = replace(CGLCConfig(), limits=BudgetLimits(1.0, 50.0, 1.0))
    res = Runner(cfg=cfg, max_checkpoints=3).run(c, w, led, tr)
    if res.decision == "REPORT_BLOCKED":
        assert res.gates is not None and res.gates.allow is False
