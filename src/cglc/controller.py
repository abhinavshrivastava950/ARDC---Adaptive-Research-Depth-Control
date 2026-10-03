"""Controller: checkpoint assembly + deterministic decision order (Sec 14.6).

Order:
 1. Conjunctive finalization gate -> ALLOW_FINALIZE
 2. User-only info needed      -> ASK_USER
 3. Infeasible under budget    -> REPORT_BLOCKED
 4. Else rank A_feas_NT by r_t -> lease for a*
Fallback when all u <= 0: VERIFY (weak/contested) > REDIRECT (stagnant
 + alternative) > shortest safe lease > ASK/REPORT_BLOCKED.

ALLOW_FINALIZE / ASK_USER / REPORT_BLOCKED never compete in the r_t
ranking; they are selected by their own conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import budget as B
from . import scoring as S
from .config import CGLCConfig
from .gates import GateSnapshot
from .guards import deterministic_guards
from .leases import Lease

A_NT = ("CONTINUE", "VERIFY", "REDIRECT")


@dataclass
class RankedAction:
    action: str
    u: float
    r: float
    d: float
    feasible: bool


def rank_actions(
    features: Dict[str, S.ActionFeatures],
    cfg: CGLCConfig,
    rho: float,
    remaining: dict,
    allowed_classes: List[str],
    blockers: List[str],
) -> Dict[str, RankedAction]:
    out: Dict[str, RankedAction] = {}
    bc = cfg.budget_control
    for a, f in features.items():
        feas = a in A_NT and deterministic_guards(blockers, remaining, a, allowed_classes)
        Pi = B.action_pressure(bc.lambda_b, rho, f.d)
        f2 = S.ActionFeatures(delta=f.delta, V=f.V, R=f.R, G=f.G, L=f.L, d=f.d, Pi=Pi)
        u = S.utility(f2, bc.lambda_v, bc.lambda_r, bc.lambda_g, bc.lambda_l)
        r = S.value_per_budget(u, f.d, bc.epsilon)
        out[a] = RankedAction(action=a, u=u, r=r, d=f.d, feasible=feas)
    return out


def decide(
    gates: GateSnapshot,
    needs_user: bool,
    infeasible: bool,
    ranked: Dict[str, RankedAction],
    weak_or_contested: bool = False,
    stagnant: bool = False,
    has_alternative: bool = False,
) -> Tuple[str, List[str]]:
    """Return (decision, rejected_alternatives)."""
    if gates.allow:
        rej = [a for a in A_NT if a not in ("ALLOW_FINALIZE",)]
        return "ALLOW_FINALIZE", rej
    if needs_user:
        return "ASK_USER", [a for a in A_NT]
    if infeasible:
        return "REPORT_BLOCKED", [a for a in A_NT]
    feas = {a: ra for a, ra in ranked.items() if ra.feasible}
    if feas:
        best = max(feas.values(), key=lambda x: (x.r, x.u))
        if best.u > 0:
            return best.action, [a for a in feas if a != best.action]
    # Fallback: never finalize merely because continuation value is low.
    if weak_or_contested and ("VERIFY" in ranked):
        return "VERIFY", [a for a in A_NT if a != "VERIFY"]
    if stagnant and has_alternative:
        return "REDIRECT", [a for a in A_NT if a != "REDIRECT"]
    if feas:
        best = max(feas.values(), key=lambda x: (x.u, x.r))
        return best.action, [a for a in feas if a != best.action]
    if needs_user:
        return "ASK_USER", list(A_NT)
    return "REPORT_BLOCKED", list(A_NT)


def lease_category_for(decision: str, near_final: bool = False,
                       early_stage: bool = False) -> str:
    if decision == "VERIFY" or near_final:
        return "SHORT"
    if decision == "REDIRECT":
        return "SHORT"
    if early_stage:
        return "EXTENDED"
    return "STANDARD"
