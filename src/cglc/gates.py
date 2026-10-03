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

    def to_dict(self) -> dict:
        """Per-gate booleans for the audit record (Sec 7.5)."""
        return {"allow": self.allow, "reasons": list(self.reasons),
                "C_terminal": self.C_terminal, "C_process": self.C_process,
                "C_evidence": self.C_evidence, "C_answer": self.C_answer,
                "C_blocker": self.C_blocker}

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


@dataclass
class Authorization:
    """Finalization authorizer output (Table 9).

    Separation of proposal from authorization (ECT): the worker may propose
    completion; only this authorizer permits it, via the conjunctive gate.
    """

    allowed: bool
    certificate: dict | None = None  # contract snapshot + receipts iff allowed
    reasons: list = None  # rejection reasons iff denied

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []


class FinalizationAuthorizer:
    """Applies the conjunctive gate and records authorization evidence.

    Output: ALLOW_FINALIZE certificate or rejection reasons (Table 9).
    Per §7.5 an authorization additionally stores the exact contract snapshot
    and the evidence receipts that supported it.
    """

    def authorize(self, gates: GateSnapshot, contract, ledger,
                  checkpoint_id: int) -> Authorization:
        if gates.allow:
            return Authorization(
                allowed=True,
                certificate={
                    "contract": contract.to_dict(),
                    "receipts": {
                        oid: [
                            {"receipt_id": r.receipt_id, "source_id": r.source_id,
                             "span_id": r.span_id, "proposition": r.proposition,
                             "relation": r.relation, "strength": r.strength}
                            for r in rs
                        ]
                        for oid, rs in ledger.receipts.items()
                    },
                    "checkpoint_id": checkpoint_id,
                },
            )
        return Authorization(allowed=False, reasons=list(gates.reasons))
