"""Audit record (Sec 7.5): every checkpoint emits a compact, reproducible
decision record; finalization additionally stores the contract snapshot
and supporting receipts; REPORT_BLOCKED stores failed gates + cause."""
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

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
