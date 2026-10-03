"""Work leases (Sec 5.6, Table 8) + guard plane (Sec 5, Table 7).

Lease = bounded authorization for ONE coarse intent. Ends at checkpoint;
never permission to continue indefinitely. Finalization / blocker /
contradiction / persistent stall ends any lease early.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid

SHORT, STANDARD, EXTENDED = 1, 3, 5


@dataclass
class Lease:
    lease_id: str
    intent: str  # CONTINUE | VERIFY | REDIRECT coarse intent label
    target_gap_ids: List[str] = field(default_factory=list)
    allowed_action_classes: List[str] = field(default_factory=list)
    action_cap: int = 3
    budget_cap: Dict[str, float] = field(default_factory=dict)
    expected_progress_test: str = ""
    actions_used: int = 0
    live: bool = True

    @staticmethod
    def make(intent: str, category: str = "STANDARD",
             target_gaps: List[str] | None = None,
             allowed: List[str] | None = None,
             expected: str = "") -> "Lease":
        caps = {"SHORT": SHORT, "STANDARD": STANDARD, "EXTENDED": EXTENDED}
        return Lease(
            lease_id=f"L-{uuid.uuid4().hex[:8]}",
            intent=intent,
            target_gap_ids=list(target_gaps or []),
            allowed_action_classes=list(
                allowed or ["SEARCH", "READ", "VERIFY", "ANSWER"]
            ),
            action_cap=caps[category.upper()],
            expected_progress_test=expected,
        )

    @property
    def expired(self) -> bool:
        return self.actions_used >= self.action_cap


def truncate_observation(text: str, tau_len: int = 3000) -> tuple[str, bool]:
    """SupervisorAgent tau_len screen: cheap noise filter (not sufficiency)."""
    if len(text) <= tau_len:
        return text, False
    return text[:tau_len] + "\n...[truncated by guard tau_len]", True


def loop_detected(action_texts: List[str], tau_loop: int = 3) -> bool:
    """True when the last tau_loop substantive actions are near-identical."""
    if len(action_texts) < tau_loop or tau_loop <= 1:
        return False
    tail = [a.strip().lower() for a in action_texts[-tau_loop:]]
    return all(t == tail[0] for t in tail)


def should_checkpoint_step(count_since_check: int, tau_step: int) -> bool:
    return count_since_check >= tau_step
