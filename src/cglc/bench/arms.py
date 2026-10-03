"""The Sec 11.1 comparison ladder as runnable arms.

Baselines (no controller) and controller variants all share the same worker
class, model and corpus, so differences come from control policy only.

  uncontrolled_worker          worker acts until it proposes to finish
  fixed_shallow / fixed_deep   exactly 1 / 5 worker actions, then answer
  generic_prompt               uncontrolled + "be thorough but concise"
  arch3_fixed_checkpoints      checkpoints every 3 actions, rule policy, no
                               scoring (Architecture-3-style approximation)
  cglc_no_stall_trigger        full CGLC with the stall trigger disabled
  cglc_rule_only               full CGLC with ordinal rules instead of scoring
  cglc_full                    full CGLC
  always_on_judge_upper_bound  full CGLC but a checkpoint after every action
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, Optional

from ..config import BudgetLimits, DEFAULT_CONFIG
from ..evaluate.metrics import ABLATIONS, no_stall_config
from ..judge import LLMJudge
from ..ledger import EvidenceLedger
from ..llm import LLMClient
from ..runner import Runner
from ..trace import Trace
from ..worker.adapter import WorkerAdapter
from ..worker.llm_worker import LLMDocumentWorker
from . import metrics as M
from .contracts import make_contract
from .datasets import BenchItem

ARMS = list(ABLATIONS)
BASELINES = {"uncontrolled_worker", "fixed_shallow", "fixed_deep", "generic_prompt"}
FIXED_SHALLOW, FIXED_DEEP, UNCONTROLLED_MAX = 1, 5, 8
GENERIC_PROMPT = "Be thorough but concise."

# arm -> Runner kwargs (policy / fixed lease) and whether the stall trigger is off
CONTROLLER_ARMS: Dict[str, Dict[str, Any]] = {
    "arch3_fixed_checkpoints": dict(policy="rules", fixed_lease=3),
    "cglc_no_stall_trigger": dict(policy="ranked", no_stall=True),
    "cglc_rule_only": dict(policy="rules"),
    "cglc_full": dict(policy="ranked"),
    "always_on_judge_upper_bound": dict(policy="ranked", fixed_lease=1),
}


def bench_config():
    return dataclasses.replace(
        DEFAULT_CONFIG,
        limits=BudgetLimits(tool_calls=12.0, tokens=80_000.0, wall_clock=900.0))


def run_item(arm: str, item: BenchItem, regime: str, llm: LLMClient,
             judge_llm: Optional[LLMClient] = None, worker_temperature: float = 0.0,
             judge_temperature: float = 0.0, max_checkpoints: int = 8) -> Dict[str, Any]:
    """Run one (arm, item); returns a flat result row. LLMConfigError propagates."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    t0 = time.time()
    trace = Trace()
    baseline = arm in BASELINES
    contract = make_contract(item, "generic" if baseline else regime)
    worker = LLMDocumentWorker(
        item.docs, contract, llm, temperature=worker_temperature,
        extra_system=GENERIC_PROMPT if arm == "generic_prompt" else "")
    adapter = WorkerAdapter(worker, trace)
    decision, checkpoints, ctrl_tokens, error = "N/A", 0, 0.0, ""
    ck_log: list = []
    judge_failures = 0

    if baseline:
        steps = {"fixed_shallow": FIXED_SHALLOW, "fixed_deep": FIXED_DEEP}.get(arm, UNCONTROLLED_MAX)
        draft = ""
        for _ in range(steps):
            res = adapter.act("CONTINUE", [], [], draft)
            draft = res.draft or draft
            if res.blocker:
                error = res.blocker
            if arm not in ("fixed_shallow", "fixed_deep") and res.propose_final:
                break
        accepted = True
    else:
        spec = CONTROLLER_ARMS[arm]
        cfg = bench_config()
        if spec.get("no_stall"):
            cfg = no_stall_config(cfg)
        ledger = EvidenceLedger([o.obligation_id for o in contract.evidence_obligations])
        judge = LLMJudge(judge_llm or llm, worker, temperature=judge_temperature)
        runner = Runner(cfg=cfg, judge=judge, max_checkpoints=max_checkpoints,
                        harvest_receipts=False, policy=spec["policy"],
                        fixed_lease=spec.get("fixed_lease"))
        res = runner.run(contract, adapter, ledger, trace)
        draft, decision = res.draft, res.decision
        judge_failures = judge.failures
        accepted = decision == "ALLOW_FINALIZE"
        checkpoints = len(res.records)
        ctrl_tokens = sum(r.controller_cost.get("tokens", 0.0) for r in res.records)
        # Why each checkpoint decided what it did: needed to diagnose false-blocks.
        ck_log = [{"decision": r.decision, "triggers": r.trigger_reasons,
                   "gate_reasons": r.gates.get("reasons", []), "note": r.note[:240]}
                  for r in res.records]

    work = [e for e in trace.events if e.action_class == "WORK"]
    step_cited = [e.detail.get("cited_docs", []) for e in work]
    cited = sorted({d for c in step_cited for d in c})
    ans = M.clean_answer(draft)
    used = trace.consumed()
    return {
        "dataset": item.dataset, "item_id": item.item_id, "arm": arm,
        "regime": "none" if baseline else regime, "qtype": item.qtype,
        "answer": ans, "draft": draft[:600], "gold": item.answers[:3],
        "em": M.exact_match(ans, item.answers), "f1": round(M.f1(ans, item.answers), 4),
        "accepted": accepted, "decision": decision,
        "steps": len(work), "checkpoints": checkpoints,
        "tool_calls": used.get("tool_calls", 0.0),
        "worker_tokens": used.get("tokens", 0.0), "controller_tokens": ctrl_tokens,
        "elapsed": round(time.time() - t0, 2),
        "cited_docs": cited, "gold_titles": item.gold_titles,
        "evidence_recall": round(M.evidence_recall(cited, item.gold_titles), 4),
        "over_steps": M.steps_after_sufficient(step_cited, item.gold_titles),
        "checkpoint_log": ck_log, "error": error[:200],
        # model calls that failed (e.g. rate limit); such rows are excluded from
        # paired comparisons because the outcome reflects infrastructure, not policy
        "infra_failures": worker.failures + judge_failures,
    }
