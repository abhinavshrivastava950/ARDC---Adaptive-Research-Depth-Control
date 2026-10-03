"""Stable task contract K (Sec 4.1, Table 6).

K = (g, H_proc, H_evid, S_soft, B_mat, A_schema, B_policy)

The contract is adapted from WebRider's intent contract but narrowed to
document-grounded work. V1 rule: hard obligations come from the
benchmark or user; automatic extraction is a separately evaluated
component (Sec 4.1, 10.1 step 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid


@dataclass
class ProcessDuty:
    duty_id: str
    description: str


@dataclass
class EvidenceObligation:
    obligation_id: str
    proposition: str
    weight: float = 1.0
    required_receipts: int = 1  # corroboration ask; ledger is per-obligation bounded


@dataclass
class TaskContract:
    contract_id: str
    goal: str
    process_duties: List[ProcessDuty] = field(default_factory=list)
    evidence_obligations: List[EvidenceObligation] = field(default_factory=list)
    soft_prefs: Dict[str, Any] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)
    answer_schema: Dict[str, Any] = field(default_factory=dict)
    budget_policy: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "benchmark/user-confirmed"
    revision: int = 1

    @staticmethod
    def create(
        goal: str,
        process_duties: List[str] | None = None,
        evidence_obligations: List[str] | None = None,
        **kw: Any,
    ) -> "TaskContract":
        duties = [
            ProcessDuty(duty_id=f"proc-{i}", description=d)
            for i, d in enumerate(process_duties or [])
        ]
        obls = [
            EvidenceObligation(obligation_id=f"ev-{i}", proposition=p)
            for i, p in enumerate(evidence_obligations or [])
        ]
        return TaskContract(
            contract_id=f"K-{uuid.uuid4().hex[:8]}",
            goal=goal,
            process_duties=duties,
            evidence_obligations=obls,
            **kw,
        )

    def obligation_weights(self) -> Dict[str, float]:
        return {o.obligation_id: o.weight for o in self.evidence_obligations}
