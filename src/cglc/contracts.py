"""Stable task contract K (Sec 4.1, Table 6).

K = (g, H_proc, H_evid, S_soft, B_mat, A_schema, B_policy)

The contract is adapted from WebRider's intent contract but narrowed to
document-grounded work. V1 rule: hard obligations come from the
benchmark or user; automatic extraction is a separately evaluated
component (Sec 4.1, 10.1 step 1).

``TaskContract.from_dict`` is the JSON entry point: the user (or benchmark)
supplies the whole tuple as one object. ``to_dict`` is its inverse: every field
that defines the contract's meaning is in it (so ``content_hash`` is honest) and
``from_dict(c.to_dict())`` rebuilds the same contract.

Where K's parts land (Sec 4.1): H_proc = ``process_duties`` (machine-checked) plus
``conduct_rules`` (shown, NOT checked); H_evid = ``evidence_obligations``; S_soft =
``soft_prefs``; B_mat = ``blockers`` (gate-closing, access-style) plus
``clarification_triggers`` (material information only the user can supply, which the
judge reports through ``needs_user`` -> ASK_USER, Sec 14.6 step 5); A_schema =
``answer_schema``; B_policy = ``budget_policy``.
"""
from __future__ import annotations

import hashlib
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
MAX_WEIGHT = 1000.0  # w_i of Eff_support / Eff_resolve (Sec 14.3): positive, finite, bounded
MAX_CONDUCT_RULES = 22  # the cglc-contract-v1 importer can produce 10 duties + 12 constraints
MAX_OBJ_CHARS = 4000  # soft_prefs / answer_schema / budget_policy (the importer allows the same)
_DUTY_KEYS = {"duty_id", "description", "check", "document", "n", "params"}
_OBL_KEYS = {"obligation_id", "proposition", "weight", "required_receipts", "conditional"}


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
    # w_i of Eff_support / Eff_resolve (Sec 14.3), read through ``obligation_weights()``. It
    # scales how much progress on this obligation counts when ranking the next lease; it never
    # changes what is required: the gate (Sec 7.1) needs every non-conditional obligation proven.
    weight: float = 1.0
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
    # B_mat, access-style: conditions that make the task impossible to complete (a login wall,
    # an unavailable page, a missing file). The judge reports one in `blocker` only if the
    # supplied documents clearly show it; a reported blocker closes the gate (C_blocker).
    blockers: List[str] = field(default_factory=list)
    answer_schema: Dict[str, Any] = field(default_factory=dict)
    budget_policy: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "benchmark/user-confirmed"
    revision: int = 1
    # Rules the answer must follow that no code can verify (conduct rules such as "browse
    # only"). Shown to the worker and the judge; reported as NOT machine-checked.
    conduct_rules: List[str] = field(default_factory=list)
    # B_mat, user-only information ("Ask if a missing budget ... would materially change the
    # recommendation"). The judge reports one through `needs_user` (-> ASK_USER, Sec 14.6
    # step 5), never through `blocker`: it cannot close the gate by itself, so an ordinary task
    # with an unstated preference is not held back forever.
    clarification_triggers: List[str] = field(default_factory=list)

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

        Strict on purpose: unknown fields (also inside duties and obligations, so a typo such
        as ``required_recepits`` cannot silently weaken a hard obligation), wrong types and
        unverifiable duties raise ValueError with a message the user can act on.
        ``provenance`` is set by the caller (the server), never trusted from the input.
        A ``contract_id`` in the input is kept (it names the contract across revisions);
        otherwise a fresh one is made. ``from_dict(c.to_dict())`` rebuilds ``c``.
        """
        if not isinstance(d, dict):
            raise ValueError(f"the contract must be a JSON object (like {{\"goal\": ...}}), but got {_kind(d)}")
        if "contract_schema" in d:  # a versioned external format: strict importer
            from .contract_import import from_cglc_v1
            return from_cglc_v1(d)
        allowed = {"goal", "process_duties", "evidence_obligations", "soft_prefs", "blockers",
                   "answer_schema", "budget_policy", "revision", "contract_id", "provenance",
                   "conduct_rules", "clarification_triggers"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            shown = sorted(allowed - {"provenance"})
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
            if size > MAX_OBJ_CHARS:
                raise ValueError(f"{name} is too large (max {MAX_OBJ_CHARS} characters of JSON)")
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

        def only_keys(x: Dict[str, Any], keys: set, name: str) -> None:
            extra = sorted(set(x) - keys)
            if extra:
                raise ValueError(f"{name} has unknown field(s) {extra}; allowed: {sorted(keys)}")

        duties: List[ProcessDuty] = []
        for i, x in enumerate(items(d.get("process_duties", []), "process_duties", 6)):
            if not isinstance(x, dict):
                raise ValueError(
                    f"process_duties[{i}] must be an object with a machine-checkable 'check' "
                    f"(one of {list(DUTY_CHECKS)}); free text cannot be verified "
                    f"(put unverifiable rules in conduct_rules)")
            only_keys(x, _DUTY_KEYS, f"process_duties[{i}]")
            check = x.get("check")
            if check not in DUTY_CHECKS:
                raise ValueError(f"process_duties[{i}].check must be one of {list(DUTY_CHECKS)}")
            given = x.get("params", {})  # the form to_dict() writes; document / n may also sit on the duty
            if not isinstance(given, dict):
                raise ValueError(f"process_duties[{i}].params must be a JSON object, but got {_kind(given)}")
            merged = dict(given)
            for k in ("document", "n"):
                if k in x:
                    if k in merged:
                        raise ValueError(f"process_duties[{i}].{k} is given twice (on the duty and in params)")
                    merged[k] = x[k]
            takes = {"cite_document": {"document"}, "min_distinct_sources": {"n"}}.get(check, set())
            if set(merged) - takes:
                raise ValueError(f"process_duties[{i}] ({check}) does not take {sorted(set(merged) - takes)}"
                                 + (f"; it takes {sorted(takes)}" if takes else ""))
            params: Dict[str, Any] = {}
            if check == "cite_document":
                params["document"] = text(merged.get("document"), f"process_duties[{i}].document", 80)
            if check == "min_distinct_sources":
                if not whole(merged.get("n"), 1):
                    raise ValueError(f"process_duties[{i}].n must be a whole number >= 1")
                params["n"] = merged["n"]
            duties.append(ProcessDuty(
                duty_id=ident(x["duty_id"] if "duty_id" in x else f"proc-{i}",
                              f"process_duties[{i}].duty_id"),
                description=text(x["description"] if "description" in x else check.replace("_", " "),
                                 f"process_duties[{i}].description", 300),
                check=check, params=params))

        obls: List[EvidenceObligation] = []
        for i, x in enumerate(items(d.get("evidence_obligations", []), "evidence_obligations", 8)):
            if isinstance(x, str):
                x = {"proposition": x}
            if not isinstance(x, dict):
                raise ValueError(f"evidence_obligations[{i}] must be a string or an object")
            only_keys(x, _OBL_KEYS, f"evidence_obligations[{i}]")
            rr = x.get("required_receipts", 1)
            if not whole(rr, 1, 5):
                raise ValueError(f"evidence_obligations[{i}].required_receipts must be 1 to 5")
            w = x.get("weight", 1.0)
            if not finite_positive(w) or w > MAX_WEIGHT:
                raise ValueError(f"evidence_obligations[{i}].weight must be a positive, finite number "
                                 f"up to {MAX_WEIGHT:g}")
            obls.append(EvidenceObligation(
                obligation_id=ident(x["obligation_id"] if "obligation_id" in x else f"ev-{i}",
                                    f"evidence_obligations[{i}].obligation_id"),
                proposition=text(x.get("proposition"), f"evidence_obligations[{i}].proposition", 400),
                weight=float(w), required_receipts=rr,
                conditional=_flag(x.get("conditional", False), f"evidence_obligations[{i}].conditional")))

        blockers = [text(b, f"blockers[{i}]", 300)
                    for i, b in enumerate(items(d.get("blockers", []), "blockers", 10))]
        clarifications = [text(b, f"clarification_triggers[{i}]", 300)
                          for i, b in enumerate(items(d.get("clarification_triggers", []),
                                                      "clarification_triggers", 10))]
        budget = small_obj(d.get("budget_policy", {}), "budget_policy")
        for k in ("max_tool_calls", "max_tokens", "max_seconds"):
            v = budget.get(k)
            if k in budget and not finite_positive(v):
                raise ValueError(f"budget_policy.{k} must be a positive, finite number")
        validate_controller_settings(budget)
        conduct = [text(r, f"conduct_rules[{i}]", 300)
                   for i, r in enumerate(items(d.get("conduct_rules", []), "conduct_rules", MAX_CONDUCT_RULES))]
        rev = d.get("revision", 1)
        if not whole(rev, 1):
            raise ValueError("revision must be a whole number >= 1")
        cid = ident(d["contract_id"], "contract_id") if "contract_id" in d else f"K-{uuid.uuid4().hex[:8]}"

        c = TaskContract(
            contract_id=cid,
            goal=text(d.get("goal"), "goal", 3000),
            process_duties=duties, evidence_obligations=obls,
            soft_prefs=small_obj(d.get("soft_prefs", {}), "soft_prefs"), blockers=blockers,
            answer_schema=small_obj(d.get("answer_schema", {}), "answer_schema"),
            budget_policy=budget, provenance=provenance, revision=rev, conduct_rules=conduct,
            clarification_triggers=clarifications)
        problems = c.validate()
        if problems:
            raise ValueError("; ".join(problems))
        return c

    def obligation_weights(self) -> Dict[str, float]:
        """w_i for Eff_support / Eff_resolve (Sec 14.3), keyed by obligation id."""
        return {o.obligation_id: o.weight for o in self.evidence_obligations}

    def content_hash(self) -> str:
        """sha256 of the canonical contract content (Table 9: 'revision ID').

        Covers what the contract *says* (goal, duties, obligations, rules, blockers,
        clarification triggers, schema, budget policy, revision) and ignores the random
        ``contract_id`` and the provenance label, so the same contract hashes the same on every
        load. The audit record stores it so a run can prove which exact contract it was checked
        against."""
        body = self.to_dict()
        body.pop("contract_id", None)
        body.pop("provenance", None)
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Contract JSON plus provenance and revision ID (Table 9). Every field that defines
        the contract's meaning is here, and ``from_dict`` accepts this shape back."""
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
            "clarification_triggers": self.clarification_triggers,
            "answer_schema": self.answer_schema,
            "budget_policy": self.budget_policy,
            "provenance": self.provenance,
            "revision": self.revision,
        }

    def validate(self) -> List[str]:
        """Structural contract check (Sec 7.4: an internally inconsistent contract stops
        execution and requests revision instead of being optimized around). The runner calls
        this before the first lease. Conservative: it flags contradictions and contracts that
        could never fail or never pass, not style. Impossibilities that depend on the supplied
        documents (a duty naming a missing document, ``n`` above the document count) are
        checked at run time by the service."""
        problems: List[str] = []

        def blank(v: Any) -> bool:
            return not isinstance(v, str) or not v.strip()

        if blank(self.goal):
            problems.append("empty goal")
        if not self.process_duties and not self.evidence_obligations:
            problems.append("no hard obligations (neither duties nor evidence)")
        ids = [d.duty_id for d in self.process_duties] + [
            o.obligation_id for o in self.evidence_obligations
        ]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            problems.append(f"duplicate obligation/duty ids {dup}")
        for d in self.process_duties:
            if d.check and d.check not in DUTY_CHECKS:
                problems.append(f"duty {d.duty_id}: unknown check {d.check!r} (use one of {list(DUTY_CHECKS)})")
            elif d.check == "cite_document" and blank(d.params.get("document")):
                problems.append(f"duty {d.duty_id}: cite_document names no document")
            elif d.check == "min_distinct_sources":
                n = d.params.get("n")
                if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                    problems.append(f"duty {d.duty_id}: min_distinct_sources needs a whole number n >= 1")
        seen: Dict[str, str] = {}
        for o in self.evidence_obligations:
            if blank(o.proposition):
                problems.append(f"obligation {o.obligation_id}: empty proposition")
                continue
            key = " ".join(o.proposition.lower().split())
            if key in seen:
                problems.append(f"obligations {seen[key]} and {o.obligation_id} state the same proposition")
            seen.setdefault(key, o.obligation_id)
            w = o.weight
            if (isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w)
                    or not 0 < w <= MAX_WEIGHT):
                problems.append(f"obligation {o.obligation_id}: weight must be a positive number up to {MAX_WEIGHT:g}")
            r = o.required_receipts
            if isinstance(r, bool) or not isinstance(r, int) or not 1 <= r <= 5:
                problems.append(f"obligation {o.obligation_id}: required_receipts must be 1 to 5")
        if (self.evidence_obligations and not self.process_duties
                and all(o.conditional for o in self.evidence_obligations)):
            problems.append("every evidence obligation is conditional and there is no process duty: "
                            "nothing would ever have to be proven")
        for name in ("blockers", "clarification_triggers", "conduct_rules"):
            if any(blank(v) for v in getattr(self, name)):
                problems.append(f"{name} has an empty entry")
        norm = lambda xs: {" ".join(x.lower().split()) for x in xs if isinstance(x, str)}  # noqa: E731
        both = sorted(norm(self.blockers) & norm(self.clarification_triggers))
        if both:
            problems.append(f"the same condition is both a blocker and a clarification trigger: {both}")
        bp = self.budget_policy
        if not isinstance(bp, dict):
            problems.append("budget_policy must be an object")
        else:
            for k in ("max_tool_calls", "max_tokens", "max_seconds"):
                v = bp.get(k)
                if k in bp and (isinstance(v, bool) or not isinstance(v, (int, float))
                                or not math.isfinite(v) or v <= 0):
                    problems.append(f"budget_policy.{k} must be a positive, finite number")
            try:
                validate_controller_settings(bp)
            except ValueError as e:
                problems.append(str(e))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            problems.append("revision must be a whole number >= 1")
        return problems
