"""Literature-backed tunable parameters for the CGLC controller.

Every default below is taken directly from its originating paper (see
PARAMETERS.md and ARCHITECTURE.md Sec 13). They are *starting points*,
not universally optimal values. Per Sec 14.7 all configured values must
be logged in the run record and evaluated via ablations on held-out
trajectories.

Groups:
 1. CGDP stagnation (tau_J, tau_U, p)
 2. Inference-Time Budget Control action scoring (lambda_b, omega, epsilon + Psi coeffs)
 3. HALT verification-aware stopping (tau, epsilon, policy)
 4. MAP-Law coverage (theta_e, theta_ev, delta, r_max) -- ablation/comparator only
 5. Stop-RAG Q(lambda) (lambda decay, T) -- deferred learned model, config only
 6. SupervisorAgent observation filtering (tau_len, tau_step, tau_loop)
 7. BATS execution temperatures (T_gen, T_select)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List
import math


@dataclass(frozen=True)
class StagnationConfig:
    """CGDP programmatic stagnation trigger (Sec 5.2 / 14.2).

    Origin: cgdp.pdf, config f_j0.6_u0.3_p2.
    Validated on LoCoMo / MuSiQue / SWE-QA-Pro across ReAct, IRCoT,
    MemGPT, Iter-RetGen; saved up to 39% tokens.
    """

    tau_J: float = 0.6  # action similarity threshold
    tau_U: float = 0.3  # unique passage rate threshold
    p: int = 2  # persistence count (consecutive rounds)
    recent_window: int = 5  # W_t: recent-action Jaccard window


@dataclass(frozen=True)
class BudgetControlConfig:
    """Inference-Time Budget Control operational scoring (Sec 5.4 / 14.4-14.5).

    Origin: Inference-Time Budget Control for LLM Search Agents.
    Fixed training-free across 48 dataset-model-budget cells
    (HotpotQA/2Wiki/MuSiQue/Bamboogle x Qwen3-32B/GPT-5.4-Mini/Qwen3.5-122B).
    Wall-clock -27.2% mean.
    """

    lambda_b: float = 0.7  # cost-penalty scale
    lambda_decomp: float = 0.14  # decomposition/bridge bonus (origin paper)
    lambda_early_ans: float = 0.18  # early-answer penalty (origin paper)
    # Resource importance weights; paper uses equal weight, sum = 1.0.
    omega: Dict[str, float] = field(
        default_factory=lambda: {"tool_calls": 1 / 3, "tokens": 1 / 3, "wall_clock": 1 / 3}
    )
    epsilon: float = 1e-5  # smoothing in denominator
    # V1 Psi structural-adjustment coefficients (Sec 14.5):
    #   Psi = lv*V + lr*R + lg*G - ll*L
    # The architecture fixes the *form*; coefficients are transparent
    # configuration (Sec 14.7). Initial values reuse the origin-paper
    # magnitudes where analogous so every term is logged and ablatable.
    lambda_v: float = 0.18  # verify-weak/contested (analogous to early-ans magnitude)
    lambda_r: float = 0.14  # redirect-opportunity (analogous to decomp bonus)
    lambda_g: float = 0.14  # gap-targeting (analogous to decomp bonus)
    lambda_l: float = 0.10  # repetition-of-stalled-trajectory penalty (tunable)
    # LOW/MEDIUM/HIGH -> numeric mapping for Delta_hat; fixed before eval.
    delta_map: Dict[str, float] = field(
        default_factory=lambda: {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
    )


@dataclass(frozen=True)
class HaltConfig:
    """HALT verification-aware stopping (Sec 7 / Table 13).

    Origin: HALT Verification-Aware Stopping for RAG agents.
    Self-Ask 3B/7B on HotpotQA/2Wiki/MuSiQue; Qwen2.5-3B verifier trained
    once on HotpotQA transferred without retraining; -20%..-45% loops.
    Used here as external evidence-relative check + bidirectional
    correction signal. Never assumes cheap pre-specified claims.
    """

    tau: float = 0.0  # verifier match-logit margin (flat optimum in [-1, 0])
    epsilon: float = 0.02  # non-inferiority margin (2pp accuracy delta)
    policy: str = "ALL_MATCH"  # every required hop must have evidence


@dataclass(frozen=True)
class CoverageConfig:
    """MAP-Law coverage / marginal gain (Table 13, Sec 9.2).

    Origin: MAP-Law, 30 synthetic labor-law pilots; 1.000 element
    coverage, rounds 7.0 -> 3.36. In CGLC this is an *ablation /
    comparator* feature, NOT a continuously maintained controller state
    (Sec 9.2 circularity rejection). Included so coverage-gain ablations
    can be reproduced.
    """

    theta_e: float = 0.85
    theta_ev: float = 0.70
    delta: float = 0.05  # marginal-gain threshold (stable in [0.00, 0.08])
    r_max: int = 7


@dataclass(frozen=True)
class StopRagConfig:
    """Stop-RAG value-based continuation (deferred learned model).

    Origin: Stop-RAG, Llama-3.1-8B + DeBERTa-v3-large Q-network on
    MuSiQue/HotpotQA/2Wiki; forward Q(lambda) +2.2..2.6 F1 over fixed
    iteration. V1 requires NO training (Sec 10.2); this config is kept
    so a learned continuation value can later be added as an ablation.
    """

    lambda_start: float = 1.0
    lambda_end: float = 0.1  # cosine decay 1.0 -> 0.1 during Q training
    T: int = 10  # max iterations

    def lam(self, progress: float) -> float:
        """Cosine decay of eligibility trace decay at progress in [0,1]."""
        q = min(1.0, max(0.0, progress))
        return self.lambda_end + 0.5 * (self.lambda_start - self.lambda_end) * (
            1 + math.cos(math.pi * q)
        )


@dataclass(frozen=True)
class ObservationFilterConfig:
    """SupervisorAgent cheap screening (guard plane, Sec 5 / Table 13).

    Origin: STOP WASTING YOUR TOKENS; tau_len=3000 optimal on
    GAIA/HumanEval/MBPP/GSM-Hard/AIME/DROP (GAIA 30.0% -> 46.7%).
    These are loop/noise screens, NOT evidence-sufficiency detectors.
    """

    tau_len: int = 3000  # excessive observation length (tokens)
    tau_step: int = 4  # inefficient-step check interval (4; 8 for GAIA-style)
    tau_step_gaia: int = 8
    tau_loop: int = 3  # loop-detection window (3; 5 for GAIA/HumanEval)
    tau_loop_gaia: int = 5


@dataclass(frozen=True)
class ExecutionTempConfig:
    """BATS execution temperatures (Sec 10.2 controller variance control).

    Origin: BATS; sweeps T in [0,1] flat ~14-16% accuracy, proving
    explicit budget tracking is robust to decoding temperature.
    """

    T_gen: float = 0.7  # diverse candidate generation
    T_select: float = 0.0  # deterministic scoring / answer commit


@dataclass(frozen=True)
class LeaseConfig:
    """Bounded work leases (Sec 5.6, Table 8, Sec 10.2)."""

    SHORT: int = 1
    STANDARD: int = 3
    EXTENDED: int = 5
    L_max: int = 5  # maximum-silence safety backstop (substantive actions)


@dataclass(frozen=True)
class BudgetLimits:
    """Initial budget limits B_j per enforced dimension (Sec 14.4)."""

    tool_calls: float = 30.0
    tokens: float = 60000.0
    wall_clock: float = 600.0  # seconds

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CGLCConfig:
    """Top-level controller configuration (Sec 10.4: config vs architecture)."""

    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    budget_control: BudgetControlConfig = field(default_factory=BudgetControlConfig)
    halt: HaltConfig = field(default_factory=HaltConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    stoprag: StopRagConfig = field(default_factory=StopRagConfig)
    obs: ObservationFilterConfig = field(default_factory=ObservationFilterConfig)
    temp: ExecutionTempConfig = field(default_factory=ExecutionTempConfig)
    lease: LeaseConfig = field(default_factory=LeaseConfig)
    limits: BudgetLimits = field(default_factory=BudgetLimits)
    # Per-obligation weights w_i default to uniform (set explicitly per contract).
    # Budget-pressure tier boundaries that force a checkpoint (Sec 5.1).
    budget_tiers: tuple = (0.5, 0.75, 0.9)

    def to_dict(self) -> dict:
        return {
            "stagnation": asdict(self.stagnation),
            "budget_control": {
                **asdict(self.budget_control),
            },
            "halt": asdict(self.halt),
            "coverage": asdict(self.coverage),
            "stoprag": asdict(self.stoprag),
            "obs": asdict(self.obs),
            "temp": asdict(self.temp),
            "lease": asdict(self.lease),
            "limits": asdict(self.limits),
            "budget_tiers": list(self.budget_tiers),
        }


DEFAULT_CONFIG = CGLCConfig()
