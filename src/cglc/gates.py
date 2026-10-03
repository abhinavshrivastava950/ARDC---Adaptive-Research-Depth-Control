"""Conjunctive finalization gate (Sec 7.1) + HALT/coverage helpers.

  ALLOW_t = C_terminal AND C_process AND C_evidence AND C_answer
            AND NOT C_blocker

A failed gate yields reasons; it never implies CONTINUE by itself --
the fallback in Sec 14.6 selects VERIFY / REDIRECT / short lease /
ASK_USER / REPORT_BLOCKED.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateSnapshot:
    C_terminal: bool = False
    C_process: bool = False
    C_evidence: bool = False
    C_answer: bool = False
    C_blocker: bool = False  # True means a material blocker is present
    reasons: list = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []

    @property
    def allow(self) -> bool:
        return bool(
            self.C_terminal and self.C_process and self.C_evidence
            and self.C_answer and not self.C_blocker
        )


def evaluate_finalization(
    has_final_candidate: bool,
    process_complete: bool,
    evidence_sufficient: bool,
    answer_conforms: bool,
    blocker_present: bool,
    detail: str = "",
) -> GateSnapshot:
    reasons: list[str] = []
    if not has_final_candidate:
        reasons.append("no final candidate (C_terminal false)")
    if not process_complete:
        reasons.append("hard process duty unmet (C_process false)")
    if not evidence_sufficient:
        reasons.append("decisive claim lacks evidence (C_evidence false)")
    if not answer_conforms:
        reasons.append("answer schema/usability violated (C_answer false)")
    if blocker_present:
        reasons.append("unresolved material blocker (C_blocker true)")
    if detail:
        reasons.append(detail)
    return GateSnapshot(
        C_terminal=has_final_candidate,
        C_process=process_complete,
        C_evidence=evidence_sufficient,
        C_answer=answer_conforms,
        C_blocker=blocker_present,
        reasons=reasons,
    )


def halt_all_match(claim_supported: list[bool], tau: float = 0.0) -> bool:
    """HALT ALL_MATCH policy stub: every required hop must have evidence.

    `tau` (default 0.0, flat optimum in [-1,0]) is the verifier
    match-logit margin; in V1 the caller thresholds verifier logits
    before passing booleans here. Kept explicit for audit/ablation.
    """
    _ = tau  # logged config; thresholding happens at verifier call site
    return bool(claim_supported) and all(claim_supported)


def coverage_ok(supported: int, total: int, theta: float) -> bool:
    if total <= 0:
        return False
    return (supported / total) >= theta
