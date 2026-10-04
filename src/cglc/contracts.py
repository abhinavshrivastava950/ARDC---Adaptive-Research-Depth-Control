"""Stable task contract K (Sec 4.1, Table 6).

K = (g, H_proc, H_evid, S_soft, B_mat, A_schema, B_policy)

The contract is adapted from WebRider's intent contract but narrowed to
document-grounded work. V1 rule: hard obligations come from the
benchmark or user; automatic extraction is a separately evaluated
component (Sec 4.1, 10.1 step 1).

``TaskContract.from_dict`` is the JSON entry point: the user (or benchmark)
supplies the whole tuple as one object.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Hard process duties must be machine-checkable (ARCHITECTURE Sec 14: the
# deterministic guarantee level enforces machine-checkable duties). Free text
# that nothing can verify is rejected instead of silently never being satisfied.
DUTY_CHECKS = ("use_every_document", "cite_document", "min_distinct_sources")
_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")


def _kind(v: Any) -> str:
    """Plain-English JSON type name for error messages."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true/false"
    if isinstance(v, (int, float)):
        return "a number"
    if isinstance(v, str):
        return "a string"
    if isinstance(v, list):
        return "a list"
    return "an object" if isinstance(v, dict) else type(v).__name__


def validate_controller_settings(budget: Dict[str, Any]) -> None:
    """Optional controller settings a contract's budget_policy may carry.

    ``structural_stall_parameters`` = {jaccard_threshold, unique_passage_rate_threshold,
    consecutive_rounds} (the CGDP stall trigger) and ``lease_action_caps`` = {SHORT,
    STANDARD, EXTENDED}. Ranges are checked here; the server may still clamp them.
    """
    sp = budget.get("structural_stall_parameters")
    if sp is not None:
        if not isinstance(sp, dict) or set(sp) != {"jaccard_threshold", "unique_passage_rate_threshold",
                                                    "consecutive_rounds"}:
            raise ValueError("budget_policy.structural_stall_parameters must be an object with exactly "
                             "jaccard_threshold, unique_passage_rate_threshold and consecutive_rounds")
        for k in ("jaccard_threshold", "unique_passage_rate_threshold"):
            v = sp[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and 0 <= v <= 1):
                raise ValueError(f"budget_policy.structural_stall_parameters.{k} must be a number from 0 to 1")
        n = sp["consecutive_rounds"]
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10:
            raise ValueError("budget_policy.structural_stall_parameters.consecutive_rounds must be 1 to 10")
    lc = budget.get("lease_action_caps")
    if lc is not None:
        if not isinstance(lc, dict) or set(lc) != {"SHORT", "STANDARD", "EXTENDED"}:
            raise ValueError("budget_policy.lease_action_caps must be an object with exactly SHORT, STANDARD, EXTENDED")
        for k, v in lc.items():
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 9:
                raise ValueError(f"budget_policy.lease_action_caps.{k} must be a whole number from 1 to 9")
        if not lc["SHORT"] <= lc["STANDARD"] <= lc["EXTENDED"]:
            raise ValueError("budget_policy.lease_action_caps must satisfy SHORT <= STANDARD <= EXTENDED")


@dataclass
class ProcessDuty:
    duty_id: str
    description: str
    check: str = ""  # one of DUTY_CHECKS ("" = verified by a caller-supplied function)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceObligation:
    obligation_id: str
    proposition: str
    weight: float = 1.0  # recorded; not used by the V1 decision yet
    required_receipts: int = 1  # distinct supporting spans needed before SUPPORTED is accepted
    # A conditional obligation ("... when relevant") may be marked not-applicable by the judge
    # when its own condition clearly does not hold for this task.
    conditional: bool = False


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
    # Rules the answer must follow that no code can verify (conduct rules such as "browse
    # only"). Shown to the worker and the judge; reported as NOT machine-checked.
    conduct_rules: List[str] = field(default_factory=list)

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

    @staticmethod
    def from_dict(d: Any, provenance: str = "user-supplied (JSON contract)") -> "TaskContract":
        """Build K = (g, H_proc, H_evid, S_soft, B_mat, A_schema, B_policy) from JSON.

        Strict on purpose: unknown fields, wrong types and unverifiable duties
        raise ValueError with a message the user can act on. ``provenance`` is
        set by the caller (the server), never trusted from the input.
        """
        if not isinstance(d, dict):
            raise ValueError(f"the contract must be a JSON object (like {{\"goal\": ...}}), but got {_kind(d)}")
        if "contract_schema" in d:  # a versioned external format: strict importer
            from .contract_import import from_cglc_v1
            return from_cglc_v1(d)
        allowed = {"goal", "process_duties", "evidence_obligations", "soft_prefs", "blockers",
                   "answer_schema", "budget_policy", "revision", "contract_id", "provenance",
                   "conduct_rules"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            shown = sorted(allowed - {"contract_id", "provenance"})
            raise ValueError(f"unknown contract field(s) {unknown}; allowed: {shown}")

        def text(v: Any, name: str, limit: int) -> str:
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{name} must be a non-empty string, but got {_kind(v)}")
            if len(v) > limit:
                raise ValueError(f"{name} is too long (max {limit} characters)")
            return v.strip()

        def items(v: Any, name: str, limit: int) -> list:
            if not isinstance(v, list):
                raise ValueError(f"{name} must be a list, but got {_kind(v)}")
            if len(v) > limit:
                raise ValueError(f"{name} has too many entries (max {limit})")
            return v

        def small_obj(v: Any, name: str) -> Dict[str, Any]:
            if not isinstance(v, dict):
                raise ValueError(f"{name} must be a JSON object, but got {_kind(v)}")
            try:
                size = len(json.dumps(v, allow_nan=False))
            except (TypeError, ValueError):
                raise ValueError(f"{name} must contain only plain JSON values (no NaN or Infinity)")
            if size > 2000:
                raise ValueError(f"{name} is too large (max 2000 characters of JSON)")
            return v

        def ident(v: Any, name: str) -> str:
            t = text(v, name, 40)
            if not _ID.match(t):
                raise ValueError(f"{name} may use only letters, digits, '_', '-' and '.' (max 40)")
            return t

        def _flag(v: Any, name: str) -> bool:
            if not isinstance(v, bool):
                raise ValueError(f"{name} must be true or false, but got {_kind(v)}")
            return v

        def finite_positive(v: Any) -> bool:
            return (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0)

        def whole(v: Any, lo: int, hi: int | None = None) -> bool:
            return (isinstance(v, int) and not isinstance(v, bool) and v >= lo
                    and (hi is None or v <= hi))

        duties: List[ProcessDuty] = []
        for i, x in enumerate(items(d.get("process_duties", []), "process_duties", 6)):
            if not isinstance(x, dict):
                raise ValueError(
                    f"process_duties[{i}] must be an object with a machine-checkable 'check' "
                    f"(one of {list(DUTY_CHECKS)}); free text cannot be verified")
            check = x.get("check")
            if check not in DUTY_CHECKS:
                raise ValueError(f"process_duties[{i}].check must be one of {list(DUTY_CHECKS)}")
            params: Dict[str, Any] = {}
            if check == "cite_document":
                params["document"] = text(x.get("document"), f"process_duties[{i}].document", 80)
            if check == "min_distinct_sources":
                if not whole(x.get("n"), 1):
                    raise ValueError(f"process_duties[{i}].n must be a whole number >= 1")
                params["n"] = x["n"]
            duties.append(ProcessDuty(
                duty_id=ident(x.get("duty_id") or f"proc-{i}", f"process_duties[{i}].duty_id"),
                description=text(x.get("description") or check.replace("_", " "),
                                 f"process_duties[{i}].description", 300),
                check=check, params=params))

        obls: List[EvidenceObligation] = []
        for i, x in enumerate(items(d.get("evidence_obligations", []), "evidence_obligations", 8)):
            if isinstance(x, str):
                x = {"proposition": x}
            if not isinstance(x, dict):
                raise ValueError(f"evidence_obligations[{i}] must be a string or an object")
            rr = x.get("required_receipts", 1)
            if not whole(rr, 1, 5):
                raise ValueError(f"evidence_obligations[{i}].required_receipts must be 1 to 5")
            w = x.get("weight", 1.0)
            if not finite_positive(w):
                raise ValueError(f"evidence_obligations[{i}].weight must be a positive, finite number")
            obls.append(EvidenceObligation(
                obligation_id=ident(x.get("obligation_id") or f"ev-{i}",
                                    f"evidence_obligations[{i}].obligation_id"),
                proposition=text(x.get("proposition"), f"evidence_obligations[{i}].proposition", 400),
                weight=float(w), required_receipts=rr,
                conditional=_flag(x.get("conditional", False), f"evidence_obligations[{i}].conditional")))

        blockers = [text(b, f"blockers[{i}]", 300)
                    for i, b in enumerate(items(d.get("blockers", []), "blockers", 10))]
        budget = small_obj(d.get("budget_policy", {}), "budget_policy")
        for k in ("max_tool_calls", "max_tokens", "max_seconds"):
            v = budget.get(k)
            if k in budget and not finite_positive(v):
                raise ValueError(f"budget_policy.{k} must be a positive, finite number")
        validate_controller_settings(budget)
        conduct = [text(r, f"conduct_rules[{i}]", 300)
                   for i, r in enumerate(items(d.get("conduct_rules", []), "conduct_rules", 10))]
        rev = d.get("revision", 1)
        if not whole(rev, 1):
            raise ValueError("revision must be a whole number >= 1")

        c = TaskContract(
            contract_id=f"K-{uuid.uuid4().hex[:8]}",
            goal=text(d.get("goal"), "goal", 3000),
            process_duties=duties, evidence_obligations=obls,
            soft_prefs=small_obj(d.get("soft_prefs", {}), "soft_prefs"), blockers=blockers,
            answer_schema=small_obj(d.get("answer_schema", {}), "answer_schema"),
            budget_policy=budget, provenance=provenance, revision=rev, conduct_rules=conduct)
        problems = c.validate()
        if problems:
            raise ValueError("; ".join(problems))
        return c

    def obligation_weights(self) -> Dict[str, float]:
        return {o.obligation_id: o.weight for o in self.evidence_obligations}

    def to_dict(self) -> Dict[str, Any]:
        """Contract JSON plus provenance and revision ID (Table 9)."""
        return {
            "contract_id": self.contract_id,
            "goal": self.goal,
            "process_duties": [
                {"duty_id": d.duty_id, "description": d.description,
                 "check": d.check, "params": d.params}
                for d in self.process_duties
            ],
            "evidence_obligations": [
                {"obligation_id": o.obligation_id, "proposition": o.proposition,
                 "weight": o.weight, "required_receipts": o.required_receipts,
                 "conditional": o.conditional}
                for o in self.evidence_obligations
            ],
            "conduct_rules": self.conduct_rules,
            "soft_prefs": self.soft_prefs,
            "blockers": self.blockers,
            "answer_schema": self.answer_schema,
            "budget_policy": self.budget_policy,
            "provenance": self.provenance,
            "revision": self.revision,
        }

    def validate(self) -> List[str]:
        """Structural contract check (§7.4: inconsistent contracts stop
        execution and request revision instead of being optimized around)."""
        problems: List[str] = []
        if not self.goal.strip():
            problems.append("empty goal")
        if not self.process_duties and not self.evidence_obligations:
            problems.append("no hard obligations (neither duties nor evidence)")
        ids = [d.duty_id for d in self.process_duties] + [
            o.obligation_id for o in self.evidence_obligations
        ]
        if len(set(ids)) != len(ids):
            problems.append("duplicate obligation/duty ids")
        return problems
