"""Live demonstration: CGLC worker under control (prints each step).

Demo 1 -- stagnation trigger: same query repeated, UPR collapses -> checkpoint.
Demo 2 -- premature finalize blocked: worker proposes completion with an
           unsupported obligation -> gate C_evidence fails -> VERIFY lease.
Demo 3 -- full comparison run with audit trail.
"""
from cglc import TaskContract, Runner, DEFAULT_CONFIG
from cglc.ledger import EvidenceLedger, EvidenceReceipt
from cglc.stagnation import StagnationTracker
from cglc.trace import Trace
from cglc.gates import evaluate_finalization
from cglc.worker.fixed_corpus import FixedCorpusWorker
from cglc.worker.adapter import WorkerAdapter

print("=" * 70)
print("DEMO 1: structural stagnation trigger (CGDP tau_J=0.6, tau_U=0.3, p=2)")
print("=" * 70)
tr = StagnationTracker()
for i, (q, chunks) in enumerate([
    ("compare architecture cost", ["docA#0", "docB#0", "docC#0"]),
    ("compare architecture cost", ["docA#0", "docB#0"]),   # repeat, 1 novel
    ("compare architecture cost", ["docA#0"]),              # repeat, 0 novel
    ("compare architecture cost", ["docA#0"]),              # repeat, 0 novel
]):
    J, U, stalled = tr.step(q, chunks)
    print(f" step {i}: J={J:.2f} (>=0.6? {J >= 0.6})  "
          f"UPR={U:.2f} (<=0.3? {U <= 0.3})  STAGNATED={stalled}")
print(" -> lease ends early, semantic checkpoint fires (never finalizes).")

print()
print("=" * 70)
print("DEMO 2: premature finalization is BLOCKED by the conjunctive gate")
print("=" * 70)
g = evaluate_finalization(
    has_final_candidate=True,   # fluent, confident draft exists
    process_complete=True,
    evidence_sufficient=False,  # decisive claim has no receipt
    answer_conforms=True,
    blocker_present=False,
)
print(" gates:", {k: getattr(g, k) for k in
      ("C_terminal", "C_process", "C_evidence", "C_answer", "C_blocker")})
print(" ALLOW =", g.allow, "| reasons:", g.reasons)
print(" -> controller issues VERIFY lease instead of finalizing.")

print()
print("=" * 70)
print("DEMO 3: full controlled comparison run (audit trail)")
print("=" * 70)
DOCS = {
    "design-A": "Architecture A uses an always-on semantic coverage layer with "
                "selective rich inspection. It supports DEEPEN VERIFY REDIRECT "
                "but coverage state is expensive.",
    "design-B": "Architecture B maintains a MAP-like research graph with "
                "requirements evidence links coverage and marginal gain. "
                "Direction awareness is strong but controller-heavy.",
    "design-C": "Architecture C logs raw trace and runs semantics at "
                "finalization fixed intervals and stalls. Hard duties sit "
                "outside effort judgment. Checkpoint timing is fixed.",
}
contract = TaskContract.create(
    goal="Choose the stronger architecture and justify with document evidence.",
    process_duties=["Inspect all three supplied designs", "Review their diagrams"],
    evidence_obligations=["Support every decisive comparison with document evidence"],
)
trace = Trace()
ledger = EvidenceLedger([o.obligation_id for o in contract.evidence_obligations])
worker = WorkerAdapter(FixedCorpusWorker(DOCS), trace)
res = Runner().run(contract, worker, ledger, trace)
print(" decision:", res.decision)
print(" trace events:", len(trace.events),
      "| budget consumed:", {k: round(v, 1) for k, v in trace.consumed().items()})
for r in res.records:
    print(f" ckpt{r.checkpoint_id} triggers={r.trigger_reasons}")
    print(f"    decision={r.decision} rejected={r.rejected}")
    print(f"    gates={r.gates} remaining={ {k: round(v,1) for k,v in r.remaining_budget.items()} }")
print(" config defaults:",
      f"tau_J={DEFAULT_CONFIG.stagnation.tau_J} tau_U={DEFAULT_CONFIG.stagnation.tau_U} "
      f"p={DEFAULT_CONFIG.stagnation.p} lambda_b={DEFAULT_CONFIG.budget_control.lambda_b} "
      f"leases=1/3/5")
