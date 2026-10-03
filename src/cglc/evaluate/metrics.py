"""Evaluation metrics (Sec 11, Table 16) + ablation harness (Sec 11.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RunMetrics:
    correctness: float = 0.0
    duty_compliance: float = 0.0
    grounding: float = 0.0
    unnecessary_continuation: int = 0
    repeated_actions: int = 0
    redirect_recovery: float = 0.0
    controller_calls: int = 0
    worker_tokens: float = 0.0
    controller_tokens: float = 0.0
    false_allow: bool = False
    false_block: bool = False


ABLATIONS = [
    "uncontrolled_worker",
    "fixed_shallow",
    "fixed_deep",
    "generic_prompt",
    "arch3_fixed_checkpoints",
    "cglc_no_stall_trigger",
    "cglc_rule_only",
    "cglc_full",
    "always_on_judge_upper_bound",
]


def summarize(records: List[Any]) -> Dict[str, Any]:
    decisions = [r.decision for r in records]
    return {
        "checkpoints": len(records),
        "decisions": decisions,
        "final": decisions[-1] if decisions else None,
    }
