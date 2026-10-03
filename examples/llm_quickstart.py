"""CGLC with a real LLM worker + LLM checkpoint judge (bring your own key).

    pip install -e ".[llm]"
    export ANTHROPIC_API_KEY=...        # PowerShell: $env:ANTHROPIC_API_KEY="..."
    python examples/llm_quickstart.py

The key is read from your environment by the Anthropic SDK; it is never
stored in this repo. Default model is claude-opus-5-5; for a cheaper run:
    CGLC_MODEL=claude-sonnet-5-5 python examples/llm_quickstart.py

This makes real, billed API calls (one per worker step + one per checkpoint).
"""
import sys

from cglc import TaskContract, Runner
from cglc.judge import LLMJudge
from cglc.ledger import EvidenceLedger
from cglc.llm import AnthropicClient, LLMConfigError
from cglc.trace import Trace
from cglc.worker.adapter import WorkerAdapter
from cglc.worker.llm_worker import LLMDocumentWorker

DOCS = {
    "design-A": "Architecture A uses an always-on semantic coverage layer with selective rich inspection. It supports DEEPEN VERIFY REDIRECT but coverage state is expensive.",
    "design-B": "Architecture B maintains a MAP-like research graph with requirements evidence links coverage and marginal gain. Direction awareness is strong but controller-heavy.",
    "design-C": "Architecture C logs raw trace and runs semantics at finalization fixed intervals and stalls. Hard duties sit outside effort judgment. Checkpoint timing is fixed.",
}

contract = TaskContract.create(
    goal="Choose the strongest architecture and justify it with document evidence.",
    process_duties=["Inspect all three supplied designs"],
    evidence_obligations=[
        "Each decisive comparison between the designs is supported by a quote from the documents",
    ],
    answer_schema={"fields": ["decision", "rationale", "limitations", "next_action"]},
)

try:
    llm = AnthropicClient()  # BYOK: ANTHROPIC_API_KEY from the environment
    trace = Trace()
    ledger = EvidenceLedger([o.obligation_id for o in contract.evidence_obligations])
    worker = LLMDocumentWorker(DOCS, contract, llm)

    def process_check():
        # Hard duty is machine-checkable: every supplied doc was actually cited.
        unmet = [] if set(DOCS) <= worker.docs_read else ["proc-0"]
        return (not unmet, unmet)

    runner = Runner(judge=LLMJudge(llm, worker), harvest_receipts=False)
    res = runner.run(contract, WorkerAdapter(worker, trace), ledger, trace,
                     process_check=process_check)
except LLMConfigError as e:
    sys.exit(f"LLM setup problem: {e}")

print("decision:", res.decision)
print("\ndraft:\n", res.draft)
print("\ncheckpoints:")
for r in res.records:
    print(f"  ckpt{r.checkpoint_id} {r.trigger_reasons} -> {r.decision} "
          f"allow={r.gates['allow']} ctrl_tokens={r.controller_cost['tokens']}")
used = trace.consumed()
print("\nworker spend:", {k: round(v, 1) for k, v in used.items()})
