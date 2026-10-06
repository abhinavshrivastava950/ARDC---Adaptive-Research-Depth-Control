"""Work leases (Sec 5.6, Table 8, Sec 6.1) + guard plane (Sec 5, Table 7).

Lease = bounded authorization for ONE coarse intent. Ends at checkpoint;
never permission to continue indefinitely. Finalization / blocker /
contradiction / persistent stall ends any lease early.

A lease carries the Sec 6.1 minimal fields: ``lease_id``, ``intent``,
``target_gap_ids``, ``allowed_action_classes`` (enforced: Sec 7.2 "action
restrictions"), ``action_cap``, ``budget_cap`` and ``expected_progress_test``.
Beside the free-text progress test it carries a structured one
(``evaluate_progress``) that the runner evaluates at the next checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping
import uuid

SHORT, STANDARD, EXTENDED = 1, 3, 5

# Action classes a lease may permit (Sec 6.1 allowed_action_classes). The
# controller names families, never exact tool calls (Sec 3.3).
SEARCH = "SEARCH"  # new retrieval / search over the corpus
READ = "READ"  # re-read passages already retrieved/cited, or read the corpus in context
VERIFY = "VERIFY"  # re-check specific claims/quotes against their cited source
ANSWER = "ANSWER"  # revise the candidate draft
ACTION_CLASSES = (SEARCH, READ, VERIFY, ANSWER)

# Default permissions per coarse intent. VERIFY keeps SEARCH so that a
# verification lease cannot loop uselessly when the source must be found again.
# An EMPTY allowed list anywhere means "unrestricted".
INTENT_ACTIONS: Dict[str, List[str]] = {
    "CONTINUE": [SEARCH, READ, ANSWER],
    "VERIFY": [SEARCH, READ, VERIFY, ANSWER],
    "REDIRECT": [SEARCH, READ, ANSWER],
}

PROGRESS_RULE = ("at least one target obligation moved up in support status or had its "
                 "contradiction resolved, or a targeted duty: gap was cleared")

_EPS = 1e-9


def action_permitted(allowed: Iterable[str] | None, action_class: str) -> bool:
    """True when ``action_class`` is allowed; an empty/None list is unrestricted."""
    allowed = list(allowed or [])
    return not allowed or action_class in allowed


def lease_budget_cap(category: str, action_cap: int, remaining: Mapping[str, float],
                     shares: Mapping[str, float]) -> Dict[str, float]:
    """Sec 6.1 ``budget_cap`` at issue time.

    tool_calls = the action cap; tokens / wall_clock = this category's share
    of what remains. A dimension absent from ``remaining`` is not capped.
    """
    share = float(shares.get(category.upper(), 1.0))
    cap: Dict[str, float] = {"tool_calls": float(action_cap)}
    for k in ("tokens", "wall_clock"):
        if k in remaining:
            cap[k] = round(share * max(0.0, float(remaining[k])), 3)
    return cap


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
    category: str = "STANDARD"  # SHORT | STANDARD | EXTENDED (FIXED under Runner.fixed_lease)
    spent: Dict[str, float] = field(default_factory=dict)
    expiry_reason: str = ""  # "" | "action_cap" | "budget_cap"
    violations: int = 0  # actions whose reported class the lease did not allow

    @staticmethod
    def make(intent: str, category: str = "STANDARD",
             target_gaps: List[str] | None = None,
             allowed: List[str] | None = None,
             expected: str = "",
             budget_cap: Dict[str, float] | None = None) -> "Lease":
        """``allowed=None`` takes the intent's default classes; ``[]`` is unrestricted."""
        caps = {"SHORT": SHORT, "STANDARD": STANDARD, "EXTENDED": EXTENDED}
        cat = category.upper()
        if allowed is None:
            allowed = INTENT_ACTIONS.get(intent, list(ACTION_CLASSES))
        return Lease(
            lease_id=f"L-{uuid.uuid4().hex[:8]}",
            intent=intent,
            target_gap_ids=list(target_gaps or []),
            allowed_action_classes=list(allowed),
            action_cap=caps[cat],
            budget_cap=dict(budget_cap or {}),
            expected_progress_test=expected,
            category=cat,
        )

    def _reason(self) -> str:
        if self.actions_used >= self.action_cap:
            return "action_cap"
        # The first action is always allowed: a cap can only end a lease after
        # something was charged against it.
        if self.actions_used > 0 and any(
                self.spent.get(k, 0.0) + _EPS >= v for k, v in self.budget_cap.items()):
            return "budget_cap"
        return ""

    @property
    def expired(self) -> bool:
        return bool(self._reason())

    def permits(self, action_class: str) -> bool:
        return action_permitted(self.allowed_action_classes, action_class)

    def charge(self, cost: Mapping[str, float] | None = None, violation: bool = False) -> str:
        """Book one worker action and its cost; returns the expiry reason ("" = still live).

        Called after the action ran, so the caps are checked after charging.
        """
        self.actions_used += 1
        for k, v in (cost or {}).items():
            self.spent[k] = self.spent.get(k, 0.0) + float(v)
        if violation:
            self.violations += 1
        self.expiry_reason = self._reason()
        return self.expiry_reason

    def evaluate_progress(self, prev_s: Mapping[str, float], prev_c: Mapping[str, int],
                          cur_s: Mapping[str, float], cur_c: Mapping[str, int],
                          weights: Mapping[str, float] | None = None,
                          unmet_duties: Iterable[str] = ()) -> Dict[str, Any]:
        """Structured progress test, evaluated at the next checkpoint (Sec 5.6, 14.3).

        Met when a TARGET obligation moved up in support status or had its
        contradiction resolved, or a targeted ``duty:`` gap was cleared
        (``unmet_duties`` = duty ids still unmet at this checkpoint).
        ``eff_support`` / ``eff_resolve`` are Sec 14.3's two quantities,
        restricted to this lease's targets and reported separately.
        """
        w = weights or {}
        progressed: List[str] = []
        eff_s = eff_r = 0.0
        for oid in self.target_gap_ids:
            if oid not in cur_s:
                continue
            ds = max(0.0, cur_s.get(oid, 0.0) - prev_s.get(oid, 0.0))
            dr = max(0, prev_c.get(oid, 0) - cur_c.get(oid, 0))
            eff_s += w.get(oid, 1.0) * ds
            eff_r += w.get(oid, 1.0) * dr
            if ds > 0 or dr > 0:
                progressed.append(oid)
        still = set(unmet_duties)
        cleared = [g for g in self.target_gap_ids
                   if g.startswith("duty:") and g[5:] not in still]
        return {
            "rule": PROGRESS_RULE,
            "met": bool(progressed or cleared),
            "targets": list(self.target_gap_ids),
            "progressed": progressed,
            "eff_support": round(eff_s, 6),
            "eff_resolve": round(eff_r, 6),
            "duties_cleared": [g[5:] for g in cleared],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Audit snapshot (Sec 6.1 fields + what the lease actually did)."""
        return {
            "lease_id": self.lease_id,
            "intent": self.intent,
            "category": self.category,
            "target_gap_ids": list(self.target_gap_ids),
            "allowed_action_classes": list(self.allowed_action_classes),
            "action_cap": self.action_cap,
            "actions_used": self.actions_used,
            "budget_cap": dict(self.budget_cap),
            "spent": {k: round(v, 6) for k, v in self.spent.items()},
            "expiry_reason": self.expiry_reason,
            "violations": self.violations,
            "expected_progress_test": self.expected_progress_test,
        }


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


@dataclass
class OverheadGuard:
    """Controller-overhead failsafe (§7.4).

    If controller cost exceeds `fraction` of total (controller + worker)
    spend, checkpoint frequency is downgraded (L_max doubled) — but the
    mandatory finalization interception is always retained.
    """

    fraction: float = 0.25
    l_max: int = 5
    downgraded: bool = False

    def observe(self, controller_tokens: float, worker_tokens: float) -> bool:
        total = controller_tokens + worker_tokens
        if total > 0 and not self.downgraded:
            if controller_tokens / total > self.fraction:
                self.l_max *= 2
                self.downgraded = True
                return True
        return False
