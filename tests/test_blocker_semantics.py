"""A worker's own 'blocker' claim triggers a checkpoint; the judge's evidence verdict decides.

B_mat reaches the judge in two kinds: access blockers (close the gate) and clarification triggers
(needs_user -> ASK_USER while the gate is closed). A 'none' blocker is no blocker."""
import json
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


# --- B_mat comes in two kinds: access blockers close the gate, clarification triggers ask the user ----------------
def _prompt(**kw):
    from cglc import TaskContract
    c = TaskContract.create("Compare A and B.", evidence_obligations=["Cite a document for each cost."], **kw)
    w = LLMDocumentWorker(DOCS, c, FakeLLM([]))
    return c, LLMJudge(FakeLLM([]), w)._user(c, EvidenceLedger(["ev-0"]), "draft")


def test_the_judge_sees_blockers_and_clarification_triggers_on_separate_lines():
    c, user = _prompt(blockers=["the page requires login"],
                      clarification_triggers=["Ask if a missing budget would change the answer."])
    lines = {l.split(" declared")[0]: l for l in user.splitlines() if " declared by the contract" in l}
    assert set(lines) == {"MATERIAL BLOCKERS", "USER-CLARIFICATION TRIGGERS"}
    assert "login" in lines["MATERIAL BLOCKERS"] and "budget" not in lines["MATERIAL BLOCKERS"]
    assert "budget" in lines["USER-CLARIFICATION TRIGGERS"] and "NOT blockers" in lines["USER-CLARIFICATION TRIGGERS"]
    _, none = _prompt()
    assert "MATERIAL BLOCKERS" not in none and "USER-CLARIFICATION" not in none


def test_required_receipts_is_shown_to_the_judge_only_when_it_matters():
    from cglc import TaskContract
    c = TaskContract.create("g", evidence_obligations=["a", "b"])
    c.evidence_obligations[1].required_receipts = 2
    w = LLMDocumentWorker(DOCS, c, FakeLLM([]))
    user = LLMJudge(FakeLLM([]), w)._user(c, EvidenceLedger(["ev-0", "ev-1"]), "d")
    shown = json.loads(user.split("EVIDENCE OBLIGATIONS: ")[1].split("\n")[0])
    assert "required_receipts" not in shown[0] and shown[1]["required_receipts"] == 2


def _judged(reply):
    w = LLMDocumentWorker(DOCS, contract(), FakeLLM([wreply(QUOTE, draft="d")]))
    w.act("CONTINUE", ["ev-0"], [], "")
    return LLMJudge(FakeLLM([reply]), w)(contract(), EvidenceLedger(["ev-0"]), "d")


def test_a_judge_that_writes_none_for_the_blocker_has_not_found_one():
    for text in ("none", "None.", "N/A", "no blocker", "  "):
        j = _judged(jreply("SUPPORTED", receipts=["A#c0"], blocker=text))
        assert not j.blocker_present and j.blocker_reason == "" and j.evidence_sufficient, text
    j = _judged(jreply("SUPPORTED", receipts=["A#c0"], blocker="the page requires login"))
    assert j.blocker_present and j.blocker_reason == "the page requires login"


def test_needs_user_reaches_the_controller_and_asks_when_the_gate_is_closed():
    res = run([wreply([], draft="guess", final=True)] * 3, [jreply("UNSEEN", needs_user=True)] * 3)
    assert res.decision == "ASK_USER"


def test_needs_user_does_not_override_a_gate_that_is_open():
    res = run([wreply(QUOTE, draft="A cited", final=True)],
              [jreply("SUPPORTED", receipts=["A#c0"], needs_user=True)])
    assert res.decision == "ALLOW_FINALIZE"
