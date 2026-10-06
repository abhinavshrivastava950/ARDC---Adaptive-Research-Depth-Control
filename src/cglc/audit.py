"""Audit record (Sec 7.5): every checkpoint emits a compact, reproducible
decision record; finalization additionally stores the contract snapshot
and supporting receipts; REPORT_BLOCKED stores failed gates + cause.

Beyond the Sec 7.5 minimum the record reports Eff_support and Eff_resolve
separately (Sec 14.3: "the controller must report these two quantities
separately"), the contract's content hash (the version the run was checked
against), the lease that ran before the checkpoint with its structured
progress test result, and the lease the decision issued."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json


@dataclass
class DecisionRecord:
    checkpoint_id: int
    contract_id: str
    contract_rev: int
    lease_id: str | None
    trigger_reasons: List[str]  # reason codes (Table 7 events)
    receipt_ids: List[str]
    gates: Dict[str, Any]
    decision: str  # decision_type (Table 10)
    rejected: List[str]
    remaining_budget: Dict[str, float]
    controller_cost: Dict[str, float]
    note: str = ""
    # Table 10 / §7.5 terminal enrichments:
    selected_lease_id: str | None = None  # lease issued for this decision
    contract_snapshot: Dict[str, Any] | None = None  # ALLOW: exact contract
    supporting_receipts: Dict[str, Any] | None = None  # ALLOW: evidence
    blocked_condition: str = ""  # REPORT_BLOCKED: cause that prevented it
    trace_len: int = 0  # trace events logged when this checkpoint ran
    # Sec 14.3: reported separately, never merged into one number.
    eff_support: float = 0.0  # sum_i w_i * max(0, s_it - s_it-)
    eff_resolve: float = 0.0  # sum_i w_i * max(0, c_it- - c_it)
    contract_hash: str = ""  # sha256 of the canonical contract content
    lease: Dict[str, Any] | None = None  # the lease that ran before this checkpoint
    lease_progress: Dict[str, Any] | None = None  # its structured progress test result
    next_lease: Dict[str, Any] | None = None  # the lease this decision issued (None if terminal)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
