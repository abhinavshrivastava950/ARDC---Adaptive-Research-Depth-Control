"""Quickstart: fixed-corpus comparison under CGLC control (Sec 10.2 scope)."""
from cglc import TaskContract, Runner, DEFAULT_CONFIG
from cglc.trace import Trace
from cglc.ledger import EvidenceLedger
from cglc.worker.fixed_corpus import FixedCorpusWorker
from cglc.worker.adapter import WorkerAdapter

DOCS = {
    "design-A": "Architecture A uses an always-on semantic coverage layer with selective rich inspection. It supports DEEPEN VERIFY REDIRECT but coverage state is expensive.",
    "design-B": "Architecture B maintains a MAP-like research graph with requirements evidence links coverage and marginal gain. Direction awareness is strong but controller-heavy.",
    "design-C": "Architecture C logs raw trace and runs semantics at finalization fixed intervals and stalls. Hard duties sit outside effort judgment. Checkpoint timing is fixed.",
}

contract = TaskContract.create(
    goal="Choose the stronger architecture and justify with document evidence.",
    process_duties=["Inspect all three supplied designs", "Review their diagrams"],
    evidence_obligations=["Support every decisive comparison with document evidence"],
)
trace = Trace()
ledger = EvidenceLedger([o.obligation_id for o in contract.evidence_obligations])
worker = WorkerAdapter(FixedCorpusWorker(DOCS), trace)
# §10.1 step 1: hard duties are benchmark/user-supplied — compliance is
# confirmed here, never inferred by the controller.
res = Runner().run(contract, worker, ledger, trace,
                   process_check=lambda: (True, []))
print("decision:", res.decision)
print("draft:", res.draft[:600])
for r in res.records:
    print(f"ckpt{r.checkpoint_id} {r.trigger_reasons} -> {r.decision} gates={r.gates}")
