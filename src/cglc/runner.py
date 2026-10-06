"""Event-driven runner: Fig. 2 state machine (Sec 5).

Loop:
  worker acts inside lease (each action's logged cost is charged to the lease;
  action cap or budget cap ends it) -> guard plane screens cheaply ->
  scheduler fires checkpoint on mandatory events ->
  checkpoint packet S_t assembled (contract + trace delta + ledger + draft) ->
  semantic judgment (pluggable) -> Eff_support / Eff_resolve + the lease's
  progress test -> gates -> rank -> next lease (sized by the Sec 5.6 policy) /
  terminal.

`judge` is the structured semantic-controller hook (Sec 10.2: structured
LLM call with low-variance schema). V1 ships a transparent rule-based
judge; swap in an LLM judge without changing the control loop.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

from . import budget as B
from . import scoring as S
from .audit import DecisionRecord
from .config import CGLCConfig, DEFAULT_CONFIG
from .contracts import TaskContract
from .controller import rank_actions, decide, decide_rules, pick_lease_category
from .gates import GateSnapshot, evaluate_finalization, FinalizationAuthorizer
from .guards import schedule_checkpoint
from .ledger import EvidenceLedger, EvidenceReceipt
from .leases import Lease, OverheadGuard, lease_budget_cap
from .stagnation import StagnationTracker
from .trace import Trace
from .worker.base import DocumentWorker


@dataclass
class Judgment:
    """Structured semantic-controller response (Table 10: gate statuses,
    open gaps, contradiction flags, progress label, direction label,
    rationale + receipt ids)."""

    evidence_sufficient: bool = False
    blocker_reason: str = ""  # human-readable cause when blocker_present
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


# Words that frame an imported obligation ("Quoted evidence for: ...", "The documents show, in a
# quote, that the answer satisfies: must ...") but say nothing about its subject. The offline
# keyword harvest must not match on them, or every obligation "matches" every passage.
_FRAME_WORDS = frozenset(
    "quoted evidence documents document quote answer satisfies must show visible supplied "
    "listing source target page name title".split())


def harvest_keys(proposition: str, limit: int = 4) -> List[str]:
    """Subject words of an obligation for the V1 offline keyword harvest (rule-judge shortcut)."""
    words = [w.strip(".,:;()\"'") for w in proposition.lower().split()]
    keys = [w for w in words if len(w) > 4 and w not in _FRAME_WORDS]
    return keys[:limit] or [w for w in words if len(w) > 4][:limit]


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
    # Sec 7.3 / 7.5: why a REPORT_BLOCKED exit happened when no checkpoint record says so
    # (the checkpoint limit ran out while the gate was still closed).
    blocked_condition: str = ""


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

    def _lease(self, intent: str, category: str, gaps,
               remaining: Optional[Dict[str, float]] = None) -> Lease:
        """Issue a lease (Sec 6.1). ``remaining`` is the budget left at issue time,
        from which the lease's ``budget_cap`` is derived (Sec 6.1)."""
        cat = category.upper()
        lease = Lease.make(intent, cat, gaps)
        # Sizes come from the config (a contract's lease_action_caps may change them).
        lease.action_cap = int(getattr(self.cfg.lease, cat, lease.action_cap))
        if self.fixed_lease:
            # Fixed-interval checkpoints (ablation arms): exactly N actions, no
            # budget caps, so the arm stays a pure action-count comparator.
            lease.action_cap = int(self.fixed_lease)
            lease.category = "FIXED"
            lease.budget_cap = {}
        else:
            lease.budget_cap = lease_budget_cap(
                cat, lease.action_cap, remaining or {}, self.cfg.lease.budget_share)
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
                note="contract revision required: " + "; ".join(problems),
                contract_hash=contract.content_hash())
            return RunResult(decision="ASK_USER", draft=draft0,
                             records=[rec], gates=None)

        stag = StagnationTracker(
            self.cfg.stagnation.tau_J, self.cfg.stagnation.tau_U,
            self.cfg.stagnation.p, self.cfg.stagnation.recent_window)
        guard = OverheadGuard(fraction=self.overhead_fraction,
                              l_max=self.cfg.lease.L_max)
        contract_hash = contract.content_hash()
        weights = contract.obligation_weights()
        # Early stage: every obligation is open and nothing is spent (Sec 5.6),
        # so the first lease is sized by the same policy as every later one.
        rem0 = self.remaining(trace)
        gaps0 = [o.obligation_id for o in contract.evidence_obligations]
        rho0 = B.remaining_pressure(rem0, self.cfg.limits.as_dict())
        lease = self._lease(
            "CONTINUE",
            pick_lease_category("CONTINUE", gaps0, len(gaps0), rho0, False, self.cfg,
                                early_stage=True),
            gaps0, rem0)
        draft = draft0
        records: List[DecisionRecord] = []
        ckpt = 0
        prev_rho_tier = 0.0
        silence = 0
        controller_tokens = 0.0
        last_gates: GateSnapshot | None = None  # the real gates of the latest checkpoint
        last_block_reason = ""
        # Ledger state after the previous checkpoint (all UNSEEN at the start):
        # Eff_support / Eff_resolve (Sec 14.3) are measured against it.
        prev_s = {o: 0.0 for o in ledger.s}
        prev_c = {o: 0 for o in ledger.c}

        while ckpt < self.max_checkpoints:
            # --- worker executes inside lease ---
            finalize_requested = False
            blocker = ""
            contradiction = False
            # Tell the worker why the last check did not pass (coarse target
            # gap + reason). It still chooses its own queries and edits.
            worker.controller_note = lease.expected_progress_test
            while lease.live and not lease.expired:
                n_before = len(trace.events)
                res = worker.act(lease.intent, lease.target_gap_ids,
                                 lease.allowed_action_classes, draft)
                draft = res.draft or draft
                # Charge the action's logged cost to the lease (Sec 6.1 budget_cap)
                # and book an action-class violation the adapter recorded (Sec 7.2).
                new_ev = trace.events[-1] if len(trace.events) > n_before else None
                lease.charge(new_ev.cost if new_ev else None,
                             violation=bool(new_ev and new_ev.status == "lease_violation"))
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
                        keys = harvest_keys(obl.proposition)
                        if any(k in o.text.lower() for k in keys):
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
                note = lease.expected_progress_test
                lease = self._lease(lease.intent, "STANDARD", lease.target_gap_ids, rem)
                lease.expected_progress_test = note
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
            # Sec 14.3: Eff_support and Eff_resolve over the interval since the
            # previous checkpoint, i.e. after the judge updated the ledger. The
            # two quantities are reported separately, never summed.
            eff_support = ledger.support_progress(prev_s, weights)
            eff_resolve = ledger.resolve_progress(prev_c, weights)
            if process_check is not None:
                proc_done, unmet = process_check()
            elif contract.process_duties:
                # No benchmark/user duty check supplied: duties stay unmet
                # and are targeted by the next lease (§5.5 step 2).
                proc_done, unmet = False, [d.duty_id for d in contract.process_duties]
            else:
                proc_done, unmet = True, []
            # The lease that just ran is tested against the structured progress
            # rule (Sec 5.6); the ledger snapshot then moves to this checkpoint.
            progress = lease.evaluate_progress(
                prev_s, prev_c, ledger.s, ledger.c, weights, unmet)
            prev_s, prev_c = ledger.snapshot()
            ran = lease.to_dict()
            gates = evaluate_finalization(
                has_final_candidate=bool(draft.strip()),
                process_complete=proc_done,
                evidence_sufficient=j.evidence_sufficient,
                answer_conforms=j.answer_conforms,
                # A worker's own 'blocker' claim triggers a checkpoint, but the judge's reading of the
                # evidence decides: if every requirement is supported, a retrieval miss reported by the
                # worker is not a reason to refuse.
                blocker_present=bool(j.blocker_present) or (bool(blocker) and not j.evidence_sufficient),
            )
            auth = self.authorizer.authorize(gates, contract, ledger, ckpt)
            last_gates, last_block_reason = gates, (j.blocker_reason or blocker)
            # Q&A Q3/Q4: a lease that acted but moved neither Eff_support nor
            # Eff_resolve must not pass as progress, so CONTINUE's Delta is
            # capped at LOW for the ranking only (flagged; never the gate, never
            # VERIFY/REDIRECT, never a finalize-request-only checkpoint, never
            # when the judge itself failed and the ledger could not move).
            feats = j.action_features
            eff_note = ""
            bc = self.cfg.budget_control
            if (self.policy == "ranked" and bc.use_eff_in_delta and "CONTINUE" in feats
                    and lease.actions_used > 0
                    and eff_support <= 1e-12 and eff_resolve <= 1e-12
                    and set(ev.reasons) != {"finalize_request"}
                    and not j.rationale.startswith("judge fail-closed")):
                low = float(bc.delta_map.get("LOW", 0.0))
                if feats["CONTINUE"].delta > low:
                    eff_note = (f" [eff-adjust: CONTINUE delta {feats['CONTINUE'].delta:g} -> {low:g}: "
                                "Eff_support=0 and Eff_resolve=0 over the last lease]")
                    feats = dict(feats)
                    feats["CONTINUE"] = dataclasses.replace(feats["CONTINUE"], delta=low)
            ranked = rank_actions(feats, self.cfg, rho, rem,
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

            note = j.rationale + eff_note
            if downgraded:
                note += " [overhead guard: checkpoint frequency downgraded]"
            if lease.violations:
                note += f" [lease violations: {lease.violations} action(s) outside the allowed classes]"
            blocked_condition = ""
            if decision == "REPORT_BLOCKED":
                dead = [k for k, v in rem.items() if v <= 0]
                if dead:
                    blocked_condition = "budget exhausted: " + ",".join(dead)
                elif blocker or j.blocker_present:
                    blocked_condition = "material blocker: " + (blocker or j.blocker_reason or "judge-flagged")
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
                    blocked_condition=blocked_condition,
                    eff_support=eff_support, eff_resolve=eff_resolve,
                    contract_hash=contract_hash, lease=ran,
                    lease_progress=progress, next_lease=None))
                return RunResult(decision=decision, draft=draft,
                                 records=records, gates=gates)
            # §5.5 step 2: unmet hard duties are targeted first.
            gaps = [f"duty:{u}" for u in unmet] + (j.open_gaps or
                    [o.obligation_id for o in contract.evidence_obligations])
            # Sec 5.6: size from the open items (evidence gaps + unmet duties),
            # budget pressure and stall state.
            cat = pick_lease_category(decision, gaps, len(unmet) + len(j.open_gaps),
                                      rho, stag.stagnated(), self.cfg)
            ran_under = lease.lease_id
            lease = self._lease(decision, cat, gaps, rem)
            if j.rationale and not j.rationale.startswith("judge fail-closed"):
                lease.expected_progress_test = j.rationale[:600]
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
                selected_lease_id=lease.lease_id,
                eff_support=eff_support, eff_resolve=eff_resolve,
                contract_hash=contract_hash, lease=ran,
                lease_progress=progress, next_lease=lease.to_dict()))

        # Checkpoint limit reached with the gate still closed (Sec 7.3): report the gate results of
        # the LAST real checkpoint, not invented ones, and say what kept it closed.
        why = f"checkpoint limit reached ({self.max_checkpoints}) with the finalization gate still closed"
        if last_gates is not None:
            gates = GateSnapshot(
                C_terminal=last_gates.C_terminal, C_process=last_gates.C_process,
                C_evidence=last_gates.C_evidence, C_answer=last_gates.C_answer,
                C_blocker=last_gates.C_blocker,
                reasons=list(last_gates.reasons) + ["max checkpoints exhausted"])
            if last_gates.C_blocker and last_block_reason:
                why += f"; material blocker: {last_block_reason}"
        else:
            gates = evaluate_finalization(bool(draft.strip()), False, False,
                                          bool(draft.strip()), False,
                                          detail="max checkpoints exhausted")
        return RunResult(decision="REPORT_BLOCKED", draft=draft,
                         records=records, gates=gates, blocked_condition=why)
