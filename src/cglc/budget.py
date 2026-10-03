"""Budget pressure + normalized action cost (Sec 14.4).

  rho_t   = 1 - min_j clip(b_jt / B_j, 0, 1)      (pressure -> 1 when scarce)
  d_t(a)  = sum_j omega_j * (g_hat_tja / B_j)      (sum omega = 1)
  Pi_t(a) = lambda_b * rho_t * d_t(a)

B_j: initial limit; b_jt: remaining; g_hat: predicted consumption.
"""
from __future__ import annotations

from typing import Dict


def clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def remaining_pressure(remaining: Dict[str, float], limits: Dict[str, float]) -> float:
    fracs = [clip01(remaining.get(k, 0.0) / limits[k]) for k in limits if limits[k] > 0]
    if not fracs:
        return 0.0
    return 1.0 - min(fracs)


def normalized_cost(predicted: Dict[str, float], limits: Dict[str, float],
                    omega: Dict[str, float]) -> float:
    tot = 0.0
    for k, Bj in limits.items():
        if Bj <= 0:
            continue
        w = omega.get(k, 0.0)
        tot += w * (predicted.get(k, 0.0) / Bj)
    return max(0.0, tot)


def action_pressure(lambda_b: float, rho: float, d: float) -> float:
    return lambda_b * rho * d
