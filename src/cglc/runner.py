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
from .controller import rank_actions, decide, decide_rules, lease_category_for
from .gates import GateSnapshot, evaluate_finalization, FinalizationAuthorizer
from .guards import schedule_checkpoint
from .ledger import EvidenceLedger, EvidenceReceipt
from .leases import Lease, OverheadGuard
from .stagnation import StagnationTracker
from .trace import Trace
from .worker.base import DocumentWorker


@dataclass
class Judgment:
    """Structured semantic-controller response (Table 10: gate statuses,
    open gaps, contradiction flags, progress label, direction label,
    rationale + receipt ids)."""

    evidence_sufficient: bool = False
    process_complete: bool = False
    answer_conforms: bool = False
    needs_user: bool = False
    infeasible: bool = False
    blocker_present: bool = False
    weak_or_contested: bool = False
    has_alternative: bool = False
    progress_label: str = "UNKNOWN"  # e.g. PROGRESSING / STALLED / COMPLETE
    direction_label: str = "UNKNOWN"  # e.g. PRODUCTIVE / VERIFY_NEEDED
    open_gaps: List[str] = field(default_factory=list)
    receipt_ids: List[str] = field(default_factory=list)
    action_features: Dict[str, S.ActionFeatures] = field(default_factory=dict)
    rationale: str = ""


def rule_judge(contract: TaskContract, ledger: EvidenceLedger, draft: str) -> Judgment:
    """Transparent V1 judge: gap = obligations still UNSEEN/PARTIAL."""
    gaps = [o.obligation_id for o in contract.evidence_obligations
            if ledger.s.get(o.obligation_id, 0.0) < 1.0]
    suff = not gaps and bool(draft.strip())
    contested = any(ledger.c.values())
    feats = {
        "CONTINUE": S.ActionFeatures(delta=0.5 if gaps else 0.0,
                                     G=1.0 if gaps else 0.0, d=0.05),
        "VERIFY": S.ActionFeatures(
            delta=0.5 if contested else 0.0,
            V=1.0 if contested else 0.0, d=0.03),
        "REDIRECT": S.ActionFeatures(delta=0.5, R=0.5, G=0.5, L=0.5, d=0.08),
    }
    return Judgment(
        evidence_sufficient=suff,
        process_complete=True,  # caller refines with real duty checks
        answer_conforms=bool(draft.strip()),
        weak_or_contested=contested,
        progress_label="COMPLETE" if suff else ("STALLED" if contested else "PROGRESSING"),
        direction_label="VERIFY_NEEDED" if contested else "PRODUCTIVE",
        open_gaps=gaps,
        action_features=feats,
        rationale="rule-judge: gaps drive CONTINUE; contradictions drive VERIFY",
    )


JudgeFn = Callable[[TaskContract, EvidenceLedger, str], Judgment]
# Benchmark/user-supplied hard-duty check (§10.1 step 1): returns
# (all_duties_complete, unmet_duty_ids). Unmet duties are targeted by the
# next lease per §5.5 step 2.
ProcessCheck = Callable[[], tuple[bool, List[str]]]


@dataclass
class RunResult:
    decision: str
    draft: str
    records: List[DecisionRecord]
    gates: GateSnapshot | None = None


class Runner:
    def __init__(self, cfg: CGLCConfig = DEFAULT_CONFIG,
                 judge: JudgeFn = rule_judge,
                 max_checkpoints: int = 12,
                 overhead_fraction: float = 0.25,
                 harvest_receipts: bool = True,
                 policy: str = "ranked",
                 fixed_lease: Optional[int] = None) -> None:
        # policy: "ranked" = value-per-cost ranking (full CGLC); "rules" =
        # ordinal rules only (the 'no action scoring' ablation, Sec 11.1).
        # fixed_lease: force every lease to this many actions (fixed-interval
        # checkpoints; 1 = always-on judge upper bound).
        if policy not in ("ranked", "rules"):
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy
        self.fixed_lease = fixed_lease
        self.cfg = cfg
        self.judge = judge
        # Keyword receipt harvesting is the rule-judge V1 shortcut. An LLM
        # judge cites its own receipts, so it should run with this off.
        self.harvest_receipts = harvest_receipts
        self.max_checkpoints = max_checkpoints
        self.overhead_fraction = overhead_fraction
        self.authorizer = FinalizationAuthorizer()

    def _lease(self, intent: str, category: str, gaps) -> Lease:
        lease = Lease.make(intent, category, gaps)
        if self.fixed_lease:
            lease.action_cap = int(self.fixed_lease)
        return lease

    def remaining(self, trace: Trace) -> Dict[str, float]:
        lim = self.cfg.limits.as_dict()
        used = trace.consumed()
        return {k: lim[k] - used.get(k, 0.0) for k in lim}

    def run(self, contract: TaskContract, worker: DocumentWorker,
            ledger: EvidenceLedger, trace: Trace,
            process_check: ProcessCheck | None = None,
            draft0: str = "") -> RunResult:
        # §7.4: inconsistent contracts stop execution and request revision.
        problems = contract.validate()
        if problems:
            rec = DecisionRecord(
                checkpoint_id=0, contract_id=contract.contract_id,
                contract_rev=contract.revision, lease_id=None,
                trigger_reasons=["contract_invalid"], receipt_ids=[],
                gates={"allow": False, "reasons": problems},
                decision="ASK_USER", rejected=[],
                remaining_budget=self.remaining(trace),
                controller_cost={"tokens": 0.0},
                note="contract revision required: " + "; ".join(problems))
            return RunResult(decision="ASK_USER", draft=draft0,
                             records=[rec], gates=None)

        stag = StagnationTracker(
            self.cfg.stagnation.tau_J, self.cfg.stagnation.tau_U,
            self.cfg.stagnation.p, self.cfg.stagnation.recent_window)
        guard = OverheadGuard(fraction=self.overhead_fraction,
                              l_max=self.cfg.lease.L_max)
        lease = self._lease("CONTINUE", "STANDARD",
                            [o.obligation_id for o in contract.evidence_obligations])
        draft = draft0
        records: List[DecisionRecord] = []
        ckpt = 0
        prev_rho_tier = 0.0
        silence = 0
        controller_tokens = 0.0

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
                for o in (res.observations if self.harvest_receipts else []):
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
                silence_hit=silence >= guard.l_max,
            )
            if not ev.run:
                # No mandatory event: extend current direction with a fresh
                # STANDARD lease (periodic inspection stays a safety net).
                lease = self._lease(lease.intent, "STANDARD", lease.target_gap_ids)
                continue

            # --- checkpoint packet S_t + semantic judgment ---
            ckpt += 1
            silence = 0
            # §7.4: downgrade checkpoint frequency on excess overhead,
            # while mandatory finalization interception is retained.
            j = self.judge(contract, ledger, draft)
            # Measured controller tokens when the judge reports them (LLM
            # judge); otherwise the V1 flat estimate.
            ctrl_cost = getattr(self.judge, "last_tokens", None)
            ctrl_cost = 500.0 if ctrl_cost is None else float(ctrl_cost)
            controller_tokens += ctrl_cost
            downgraded = guard.observe(
                controller_tokens, trace.consumed().get("tokens", 0.0))
            if process_check is not None:
                proc_done, unmet = process_check()
            elif contract.process_duties:
                # No benchmark/user duty check supplied: duties stay unmet
                # and are targeted by the next lease (§5.5 step 2).
                proc_done, unmet = False, [d.duty_id for d in contract.process_duties]
            else:
                proc_done, unmet = True, []
            gates = evaluate_finalization(
                has_final_candidate=bool(draft.strip()),
                process_complete=proc_done,
                evidence_sufficient=j.evidence_sufficient,
                answer_conforms=j.answer_conforms,
                blocker_present=bool(blocker) or j.blocker_present,
            )
            auth = self.authorizer.authorize(gates, contract, ledger, ckpt)
            ranked = rank_actions(j.action_features, self.cfg, rho, rem,
                                  ["CONTINUE", "VERIFY", "REDIRECT"],
                                  contract.blockers)
            if self.policy == "rules":
                decision, rejected = decide_rules(
                    gates, j.needs_user, j.infeasible,
                    weak_or_contested=j.weak_or_contested or contradiction,
                    stagnant=stag.stagnated(), has_alternative=j.has_alternative,
                    budget_alive=all(v > 0 for v in rem.values()))
            else:
                decision, rejected = decide(
                    gates, j.needs_user, j.infeasible, ranked,
                    weak_or_contested=j.weak_or_contested or contradiction,
                stagnant=stag.stagnated(), has_alternative=j.has_alternative)

            note = j.rationale
            if downgraded:
                note += " [overhead guard: checkpoint frequency downgraded]"
            blocked_condition = ""
            if decision == "REPORT_BLOCKED":
                dead = [k for k, v in rem.items() if v <= 0]
                if dead:
                    blocked_condition = "budget exhausted: " + ",".join(dead)
                elif blocker or j.blocker_present:
                    blocked_condition = "material blocker: " + (blocker or "judge-flagged")
                else:
                    blocked_condition = "no feasible non-terminal action"

            if decision in ("ALLOW_FINALIZE", "ASK_USER", "REPORT_BLOCKED"):
                records.append(DecisionRecord(
                    checkpoint_id=ckpt, contract_id=contract.contract_id,
                    contract_rev=contract.revision,
                    lease_id=lease.lease_id, trigger_reasons=ev.reasons,
                    receipt_ids=list(j.receipt_ids),
                    gates=gates.to_dict(),
                    decision=decision, rejected=rejected,
                    remaining_budget=dict(rem), trace_len=len(trace.events),
                    controller_cost={"tokens": ctrl_cost},
                    note=note,
                    selected_lease_id=None,
                    contract_snapshot=(auth.certificate or {}).get("contract"),
                    supporting_receipts=(auth.certificate or {}).get("receipts"),
                    blocked_condition=blocked_condition))
                return RunResult(decision=decision, draft=draft,
                                 records=records, gates=gates)
            cat = lease_category_for(decision)
            # §5.5 step 2: unmet hard duties are targeted first.
            gaps = [f"duty:{u}" for u in unmet] + (j.open_gaps or
                    [o.obligation_id for o in contract.evidence_obligations])
            ran_under = lease.lease_id
            lease = self._lease(decision, cat, gaps)
            records.append(DecisionRecord(
                checkpoint_id=ckpt, contract_id=contract.contract_id,
                contract_rev=contract.revision,
                lease_id=ran_under, trigger_reasons=ev.reasons,
                receipt_ids=list(j.receipt_ids),
                gates=gates.to_dict(),
                decision=decision, rejected=rejected,
                remaining_budget=dict(rem), trace_len=len(trace.events),
                controller_cost={"tokens": ctrl_cost},
                note=note,
                selected_lease_id=lease.lease_id))

        gates = evaluate_finalization(bool(draft.strip()), False, False,
                                      bool(draft.strip()), False,
                                      detail="max checkpoints exhausted")
        return RunResult(decision="REPORT_BLOCKED", draft=draft,
                         records=records, gates=gates)
