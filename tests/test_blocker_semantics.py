"""A worker's own 'blocker' claim triggers a checkpoint; the judge's evidence verdict decides."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_llm import DOCS, FakeLLM, contract, jreply, wreply  # noqa: E402

from cglc import Runner  # noqa: E402
from cglc.judge import LLMJudge  # noqa: E402
from cglc.ledger import EvidenceLedger  # noqa: E402
from cglc.trace import Trace  # noqa: E402
from cglc.worker.adapter import WorkerAdapter  # noqa: E402
from cglc.worker.llm_worker import LLMDocumentWorker  # noqa: E402


def run(worker_replies, judge_replies):
    c = contract()
    trace = Trace()
    w = LLMDocumentWorker(DOCS, c, FakeLLM(worker_replies))
    runner = Runner(judge=LLMJudge(FakeLLM(judge_replies), w), harvest_receipts=False)
    return runner.run(c, WorkerAdapter(w, trace), EvidenceLedger(["ev-0"]), trace,
                      process_check=lambda: (True, []))


QUOTE = [("A", "always-on semantic coverage layer")]


def test_worker_retrieval_miss_does_not_block_when_the_judge_finds_everything_supported():
    res = run([wreply(QUOTE, draft="A cited", final=True,
                      blocker="No retrieved passage mentions the second part.")],
              [jreply("SUPPORTED", receipts=["A#c0"])])
    assert res.decision == "ALLOW_FINALIZE"


def test_worker_blocker_still_stops_things_when_evidence_is_not_sufficient():
    res = run([wreply([], draft="guess", final=True, blocker="the document is missing")] * 12,
              [jreply("UNSEEN")] * 12)
    assert res.decision != "ALLOW_FINALIZE"


def test_judge_flagged_blocker_always_blocks():
    res = run([wreply(QUOTE, draft="A cited", final=True)] * 12,
              [jreply("SUPPORTED", receipts=["A#c0"], blocker="declared blocker holds")] * 12)
    assert res.decision != "ALLOW_FINALIZE"
