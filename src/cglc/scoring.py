"""Operational action score (Sec 5.4 / 14.5).

  Delta_hat in [0,1] (LOW=0, MEDIUM=0.5, HIGH=1)
  Psi       = lv*V + lr*R + lg*G - ll*L, features in [0,1]
  u_t(a)    = Delta + Psi - Pi
  r_t(a)    = max(0, u) / (d + eps)

MATHEMATICAL BOUNDARY (Sec 5.4): r_t is an operational ranking proxy,
not Bayesian VOI, not an optimality proof. Ranks feasible
non-terminal actions only; never overrides gates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionFeatures:
    delta: float  # predicted contract-relevant progress in [0,1]
    V: float = 0.0  # verify weak/contested claim
    R: float = 0.0  # move away from unproductive direction
    G: float = 0.0  # target high-priority unresolved gap
    L: float = 0.0  # risk of repeating stalled trajectory
    d: float = 0.0  # normalized predicted cost
    Pi: float = 0.0  # budget-dependent pressure


def structural_adjustment(f: ActionFeatures, lv: float, lr: float,
                          lg: float, ll: float) -> float:
    return lv * f.V + lr * f.R + lg * f.G - ll * f.L


def utility(f: ActionFeatures, lv: float, lr: float, lg: float, ll: float) -> float:
    return f.delta + structural_adjustment(f, lv, lr, lg, ll) - f.Pi


def value_per_budget(u: float, d: float, eps: float = 1e-5) -> float:
    return max(0.0, u) / (d + eps)


def delta_from_label(label: str, mapping: dict) -> float:
    return float(mapping[label.upper()])
