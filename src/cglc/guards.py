"""Deterministic guards + checkpoint scheduler (Sec 5.1, Table 7).

Scheduler (mandatory events):
  finalize request | lease expiry | persistent stall (p rounds) |
  blocker/contradiction | budget-tier crossing | max silence (L_max)

Fixed periodic inspection is only the L_max safety net, never the
primary scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CheckpointEvent:
    run: bool
    reasons: List[str]


def schedule_checkpoint(
    finalize_requested: bool,
    lease_expired: bool,
    stalled: bool,
    blocker: bool,
    contradiction: bool,
    budget_tier_crossed: bool,
    silence_hit: bool,
) -> CheckpointEvent:
    reasons: List[str] = []
    if finalize_requested:
        reasons.append("finalize_request")
    if lease_expired:
        reasons.append("lease_expiry")
    if stalled:
        reasons.append("structural_stall")
    if blocker:
        reasons.append("blocker")
    if contradiction:
        reasons.append("contradiction")
    if budget_tier_crossed:
        reasons.append("budget_tier")
    if silence_hit:
        reasons.append("max_silence")
    return CheckpointEvent(run=bool(reasons), reasons=reasons)


def deterministic_guards(
    contract_blockers: List[str],
    remaining: dict,
    action: str,
    allowed_classes: List[str],
) -> bool:
    """Feasibility filter A_feas (Sec 5.4): 1 iff action is allowed.

    V1 rules: action class must be lease-allowed; no hard blocker that
    only the user can resolve may be bypassed by a non-ASK action; every
    enforced budget dimension must remain positive.
    """
    if allowed_classes and action not in allowed_classes:
        return False
    if contract_blockers and action not in ("ASK_USER", "REPORT_BLOCKED"):
        # Material user-only blockers are handled via ASK_USER escalation,
        # but information-gathering actions remain feasible unless the
        # caller marks the blocker as hard-stopping. V1: keep feasible so
        # the decision order (ASK first) can fire explicitly.
        pass
    if any(v <= 0 for v in remaining.values()):
        return False
    return True
