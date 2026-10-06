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
from typing import Dict, List, Sequence, Tuple

from . import budget as B
from . import scoring as S
from .config import CGLCConfig, DEFAULT_CONFIG
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


def decide_rules(
    gates: GateSnapshot,
    needs_user: bool,
    infeasible: bool,
    weak_or_contested: bool = False,
    stagnant: bool = False,
    has_alternative: bool = False,
    budget_alive: bool = True,
) -> Tuple[str, List[str]]:
    """Rule-only policy: the 'no action scoring' ablation (Sec 11.1).

    Same terminal logic as ``decide``; the non-terminal choice is purely
    ordinal (contested -> VERIFY, stalled with an alternative -> REDIRECT,
    else CONTINUE) with no value-per-cost ranking.
    """
    if gates.allow:
        return "ALLOW_FINALIZE", list(A_NT)
    if needs_user:
        return "ASK_USER", list(A_NT)
    if infeasible or not budget_alive:
        return "REPORT_BLOCKED", list(A_NT)
    if weak_or_contested:
        return "VERIFY", [a for a in A_NT if a != "VERIFY"]
    if stagnant and has_alternative:
        return "REDIRECT", [a for a in A_NT if a != "REDIRECT"]
    return "CONTINUE", [a for a in A_NT if a != "CONTINUE"]


def lease_category_for(decision: str, near_final: bool = False,
                       early_stage: bool = False) -> str:
    """Flag-driven category (kept for callers; the runner uses ``pick_lease_category``)."""
    if decision == "VERIFY" or near_final:
        return "SHORT"
    if decision == "REDIRECT":
        return "SHORT"
    if early_stage:
        return "EXTENDED"
    return "STANDARD"


def pick_lease_category(decision: str,
                        open_gaps: Sequence[str] | int = (),
                        n_open_total: int | None = None,
                        rho: float = 0.0,
                        stalled: bool = False,
                        cfg: CGLCConfig = DEFAULT_CONFIG,
                        early_stage: bool = False) -> str:
    """Lease size policy (Sec 5.6): a pure function of logged inputs and thresholds.

      VERIFY, REDIRECT                      -> SHORT (verification, redirection)
      CONTINUE, open <= short_max_gaps      -> SHORT (near-final)
      CONTINUE, rho >= short_min_pressure   -> SHORT (high budget pressure)
      CONTINUE, open >= extended_min_gaps, rho <= extended_max_pressure,
                not stalled                 -> EXTENDED (early stage, many
                                               independent items, low pressure)
      otherwise                             -> STANDARD (default)

    ``open`` is ``n_open_total`` (open evidence gaps plus unmet process duties)
    when given, else the number of ``open_gaps`` (a list of gap ids or a count).
    ``early_stage`` marks the first lease, issued before any work exists: nothing
    can be "near-final" yet, so the near-final rule is skipped (a one-obligation
    task still gets a STANDARD first lease, not a SHORT one). Thresholds live in
    ``cfg.lease`` (V1 conventions, tunable). EXTENDED never removes the
    maximum-silence backstop (L_max stays active).
    """
    if decision in ("VERIFY", "REDIRECT"):
        return "SHORT"
    lc = cfg.lease
    if n_open_total is not None:
        n = int(n_open_total)
    else:
        n = open_gaps if isinstance(open_gaps, int) else len(open_gaps)
    if (n <= lc.short_max_gaps and not early_stage) or rho >= lc.short_min_pressure:
        return "SHORT"
    if n >= lc.extended_min_gaps and rho <= lc.extended_max_pressure and not stalled:
        return "EXTENDED"
    return "STANDARD"
