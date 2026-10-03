"""Bounded evidence ledger (Sec 4.3, 14.3, Table 10).

Per obligation i: support status s_it in {0, 1/2, 1} = UNSEEN/PARTIAL/
SUPPORTED plus separate contradiction flag c_it in {0,1}. CONTESTED is
*not* a support value (Sec 14.3 correction): an obligation may be
SUPPORTED yet still carry a material contradiction.

Ledger is bounded *per obligation*: strongest direct receipt + small
corroboration set + material contradictions. Raw artifacts stay in the
trace store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

UNSEEN, PARTIAL, SUPPORTED = 0.0, 0.5, 1.0
MAX_CORROBORATING = 2


@dataclass
class EvidenceReceipt:
    receipt_id: str
    source_id: str
    span_id: str
    proposition: str
    obligation_ids: List[str]
    relation: str  # supports | contradicts | context
    strength: float = 1.0  # in [0,1]
    checkpoint_id: Optional[int] = None


class EvidenceLedger:
    def __init__(self, obligation_ids: List[str]) -> None:
        self.s: Dict[str, float] = {o: UNSEEN for o in obligation_ids}
        self.c: Dict[str, int] = {o: 0 for o in obligation_ids}
        self.receipts: Dict[str, List[EvidenceReceipt]] = {
            o: [] for o in obligation_ids
        }

    def add_receipt(self, r: EvidenceReceipt) -> None:
        for oid in r.obligation_ids:
            if oid not in self.receipts:
                self.receipts[oid] = []
                self.s.setdefault(oid, UNSEEN)
                self.c.setdefault(oid, 0)
            bucket = self.receipts[oid]
            if r.relation == "contradicts":
                bucket.append(r)
                self.c[oid] = 1
            elif r.relation == "supports":
                # keep strongest direct receipt first, then corroboration cap
                bucket.append(r)
                bucket.sort(key=lambda x: -x.strength)
                supports = [x for x in bucket if x.relation == "supports"]
                others = [x for x in bucket if x.relation != "supports"]
                bucket[:] = supports[: 1 + MAX_CORROBORATING] + others
                best = supports[0].strength if supports else 0.0
                if best >= 0.8 or len(supports) >= 2:
                    self.s[oid] = SUPPORTED
                elif supports:
                    self.s[oid] = PARTIAL
            else:
                bucket.append(r)
                if self.s[oid] == UNSEEN:
                    self.s[oid] = PARTIAL

    def resolve_contradiction(self, obligation_id: str) -> None:
        self.c[obligation_id] = 0

    def support_progress(
        self, prev: Dict[str, float], weights: Dict[str, float] | None = None
    ) -> float:
        """Eff_support = sum_i w_i * max(0, s_it - s_it-). (Sec 14.3)"""
        tot = 0.0
        for oid, cur in self.s.items():
            w = (weights or {}).get(oid, 1.0)
            tot += w * max(0.0, cur - prev.get(oid, UNSEEN))
        return tot

    def resolve_progress(
        self, prev_c: Dict[str, int], weights: Dict[str, float] | None = None
    ) -> float:
        """Eff_resolve = sum_i w_i * max(0, c_it- - c_it). (Sec 14.3)"""
        tot = 0.0
        for oid, cur in self.c.items():
            w = (weights or {}).get(oid, 1.0)
            tot += w * max(0, prev_c.get(oid, 0) - cur)
        return tot

    def snapshot(self) -> tuple[Dict[str, float], Dict[str, int]]:
        return dict(self.s), dict(self.c)
