"""Evaluation metrics (Sec 11, Table 16) + ablation harness (Sec 11.1)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Dict, Any


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
    """Per-run summary. Eff_support and Eff_resolve stay separate (Sec 14.3)."""
    decisions = [r.decision for r in records]
    tested = [r.lease_progress for r in records if getattr(r, "lease_progress", None)]
    return {
        "checkpoints": len(records),
        "decisions": decisions,
        "final": decisions[-1] if decisions else None,
        "eff_support_total": round(sum(getattr(r, "eff_support", 0.0) for r in records), 6),
        "eff_resolve_total": round(sum(getattr(r, "eff_resolve", 0.0) for r in records), 6),
        "lease_progress_tested": len(tested),
        "lease_progress_met": sum(1 for p in tested if p.get("met")),
        "lease_violations": sum((getattr(r, "lease", None) or {}).get("violations", 0)
                                for r in records),
    }


def no_stall_config(cfg):
    """Ablation variant: structural stall trigger disabled (§11.1).

    tau_J above any achievable Jaccard value and tau_U below any
    achievable UPR make Stagnated_t unsatisfiable; all other
    parameters are untouched.
    """
    return replace(
        cfg,
        stagnation=replace(cfg.stagnation, tau_J=2.0, tau_U=-1.0),
    )


def run_ablations(make_case: Callable[[], tuple], variants: List[tuple]) -> List[Dict[str, Any]]:
    """Run named (config, judge, process_check) variants over fresh cases.

    Each variant is (name, cfg, judge, process_check); make_case() returns
    (contract, worker, ledger, trace). Returns one summary row per variant
    for the ablation report (Table 9: offline evaluator output).
    """
    from ..runner import Runner  # local import: evaluate is downstream of core

    rows: List[Dict[str, Any]] = []
    for name, cfg, judge, process_check in variants:
        contract, worker, ledger, trace = make_case()
        kw: Dict[str, Any] = {}
        if judge is not None:
            kw["judge"] = judge
        res = Runner(cfg=cfg, **kw).run(
            contract, worker, ledger, trace, process_check=process_check)
        row = {"variant": name, **summarize(res.records)}
        rows.append(row)
    return rows
