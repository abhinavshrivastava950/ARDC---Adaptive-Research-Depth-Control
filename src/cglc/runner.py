"""Event-driven runner: Fig. 2 state machine (Sec 5).

Loop:
  worker acts inside lease -> guard plane screens cheaply ->
  scheduler fires checkpoint on mandatory events ->
  checkpoint packet S_t assembled (contract + trace delta + ledger + draft) ->
  semantic judgment (pluggable) -> gates -> rank -> next lease / terminal.

`judge` is the structured semantic-controller hook (Sec 10.2: structured
LLM call with low-variance schema). V1 ships a transparent rule-based
judge; swap in an LLM judge without changing the control loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

from . import budget as B
from . import scoring as S
from .audit import DecisionRecord
from .config import CGLCConfig, DEFAULT_CONFIG
from .contracts import TaskContract
from .controller import rank_actions, decide, lease_category_for
from .gates import GateSnapshot, evaluate_finalization
from .guards import schedule_checkpoint
from .ledger import EvidenceLedger, EvidenceReceipt
from .leases import Lease
from .stagnation import StagnationTracker
from .trace import Trace
from .worker.base import DocumentWorker


@dataclass
class Judgment:
    """Structured semantic-controller response (cited receipts required)."""

    evidence_sufficient: bool = False
    process_complete: bool = False
    answer_conforms: bool = False
    needs_user: bool = False
    infeasible: bool = False
    blocker_present: bool = False
    weak_or_contested: bool = False
    has_alternative: bool = False
    open_gaps: List[str] = field(default_factory=list)
    receipt_ids: List[str] = field(default_factory=list)
    action_features: Dict[str, S.ActionFeatures] = field(default_factory=dict)
    rationale: str = ""


def rule_judge(contract: TaskContract, ledger: EvidenceLedger, draft: str) -> Judgment:
    """Transparent V1 judge: gap = obligations still UNSEEN/PARTIAL."""
    gaps = [o.obligation_id for o in contract.evidence_obligations
            if ledger.s.get(o.obligation_id, 0.0) < 1.0]
    suff = not gaps and bool(draft.strip())
    feats = {
        "CONTINUE": S.ActionFeatures(delta=0.5 if gaps else 0.0,
                                     G=1.0 if gaps else 0.0, d=0.05),
        "VERIFY": S.ActionFeatures(
            delta=0.5 if any(ledger.c.values()) else 0.0,
            V=1.0 if any(ledger.c.values()) else 0.0, d=0.03),
        "REDIRECT": S.ActionFeatures(delta=0.5, R=0.5, G=0.5, L=0.5, d=0.08),
    }
    return Judgment(
        evidence_sufficient=suff,
        process_complete=True,  # caller refines with real duty checks
        answer_conforms=bool(draft.strip()),
        open_gaps=gaps,
        action_features=feats,
        rationale="rule-judge: gaps drive CONTINUE; contradictions drive VERIFY",
    )


JudgeFn = Callable[[TaskContract, EvidenceLedger, str], Judgment]


@dataclass
class RunResult:
    decision: str
    draft: str
    records: List[DecisionRecord]
    gates: GateSnapshot | None = None


class Runner:
    def __init__(self, cfg: CGLCConfig = DEFAULT_CONFIG,
                 judge: JudgeFn = rule_judge,
                 max_checkpoints: int = 12) -> None:
        self.cfg = cfg
        self.judge = judge
        self.max_checkpoints = max_checkpoints

    def remaining(self, trace: Trace) -> Dict[str, float]:
        lim = self.cfg.limits.as_dict()
        used = trace.consumed()
        return {k: lim[k] - used.get(k, 0.0) for k in lim}

    def run(self, contract: TaskContract, worker: DocumentWorker,
            ledger: EvidenceLedger, trace: Trace,
            process_complete_fn: Callable[[], bool] | None = None,
            draft0: str = "") -> RunResult:
        stag = StagnationTracker(
            self.cfg.stagnation.tau_J, self.cfg.stagnation.tau_U,
            self.cfg.stagnation.p, self.cfg.stagnation.recent_window)
        lease = Lease.make("CONTINUE", "STANDARD",
                           [o.obligation_id for o in contract.evidence_obligations])
        draft = draft0
        records: List[DecisionRecord] = []
        ckpt = 0
        prev_rho_tier = 0.0
        silence = 0

        while ckpt < self.max_checkpoints:
            # --- worker executes inside lease ---
            finalize_requested = False
            blocker = ""
            contradiction = False
            lease.actions_used = 0
            while lease.live and not lease.expired:
                res = worker.act(lease.intent, lease.target_gap_ids,
                                 lease.allowed_action_classes, draft)
                draft = res.draft or draft
                lease.actions_used += 1
                silence += 1
                # feed stagnation tracker from last trace event
                if trace.events:
                    last = trace.events[-1]
                    _J, _U, stalled_now = stag.step(
                        last.action_text, last.chunk_ids)
                else:
                    stalled_now = False
                # naive receipt harvesting: any observation mentioning an
                # obligation proposition counts as weak PARTIAL support.
                for o in res.observations:
                    for obl in contract.evidence_obligations:
                        keys = [w for w in obl.proposition.lower().split() if len(w) > 4]
                        if any(k in o.text.lower() for k in keys[:4]):
                            ledger.add_receipt(EvidenceReceipt(
                                receipt_id=f"R-{len(trace.events)}-{obl.obligation_id}",
                                source_id=o.source_id, span_id=o.span_id,
                                proposition=obl.proposition,
                                obligation_ids=[obl.obligation_id],
                                relation="supports", strength=0.6,
                                checkpoint_id=ckpt))
                if res.propose_final:
                    finalize_requested = True
                    break
                if res.blocker:
                    blocker = res.blocker
                    break
                if res.contradiction:
                    contradiction = True
                    break
                if stag.stagnated():
                    break

            # --- guard plane: scheduler ---
            rem = self.remaining(trace)
            lim = self.cfg.limits.as_dict()
            rho = B.remaining_pressure(rem, lim)
            tier = sum(1 for t in self.cfg.budget_tiers if rho >= t)
            tier_crossed = tier > prev_rho_tier
            prev_rho_tier = tier
            ev = schedule_checkpoint(
                finalize_requested=finalize_requested,
                lease_expired=lease.expired,
                stalled=stag.stagnated(),
                blocker=bool(blocker),
                contradiction=contradiction,
                budget_tier_crossed=tier_crossed,
                silence_hit=silence >= self.cfg.lease.L_max,
            )
            if not ev.run:
                # No mandatory event: extend current direction with a fresh
                # STANDARD lease (periodic inspection stays a safety net).
                lease = Lease.make(lease.intent, "STANDARD", lease.target_gap_ids)
                continue

            # --- checkpoint packet S_t + semantic judgment ---
            ckpt += 1
            silence = 0
            j = self.judge(contract, ledger, draft)
            proc_done = process_complete_fn() if process_complete_fn else j.process_complete
            gates = evaluate_finalization(
                has_final_candidate=bool(draft.strip()),
                process_complete=proc_done,
                evidence_sufficient=j.evidence_sufficient,
                answer_conforms=j.answer_conforms,
                blocker_present=bool(blocker) or j.blocker_present,
            )
            ranked = rank_actions(j.action_features, self.cfg, rho, rem,
                                  ["CONTINUE", "VERIFY", "REDIRECT"],
                                  contract.blockers)
            decision, rejected = decide(
                gates, j.needs_user, j.infeasible, ranked,
                weak_or_contested=j.weak_or_contested or contradiction,
                stagnant=stag.stagnated(), has_alternative=j.has_alternative)
            records.append(DecisionRecord(
                checkpoint_id=ckpt, contract_id=contract.contract_id,
                contract_rev=contract.revision,
                lease_id=lease.lease_id, trigger_reasons=ev.reasons,
                receipt_ids=list(j.receipt_ids),
                gates={"allow": gates.allow, "reasons": gates.reasons},
                decision=decision, rejected=rejected,
                remaining_budget=dict(rem),
                controller_cost={"tokens": 500.0},
                note=j.rationale))

            if decision in ("ALLOW_FINALIZE", "ASK_USER", "REPORT_BLOCKED"):
                return RunResult(decision=decision, draft=draft,
                                 records=records, gates=gates)
            cat = lease_category_for(decision)
            gaps = j.open_gaps or [o.obligation_id for o in contract.evidence_obligations]
            lease = Lease.make(decision, cat, gaps)
            for _r in records:
                pass

        gates = evaluate_finalization(bool(draft.strip()), False, False,
                                      bool(draft.strip()), False,
                                      detail="max checkpoints exhausted")
        return RunResult(decision="REPORT_BLOCKED", draft=draft,
                         records=records, gates=gates)
