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
    trigger_reasons: List[str]
    receipt_ids: List[str]
    gates: Dict[str, Any]
    decision: str
    rejected: List[str]
    remaining_budget: Dict[str, float]
    controller_cost: Dict[str, float]
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
