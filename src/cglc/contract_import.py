"""Importer for the ``cglc-contract-v1`` contract format (e.g. the WebRider tasks).

That format describes a *web-browsing* task. This project is document-grounded, so the
import is an explicit, documented mapping and never pretends more than it does:

  goal                  -> goal
  evidence_obligations  -> evidence obligations ("Quoted evidence for: ...");
                           wording that is conditional ("when relevant", "if visible")
                           becomes a conditional obligation the judge may mark not applicable
  hard_task_constraints -> constraints of the form "must show / be / address / use /
                           distinguish ..." become evidence obligations (each needs a quote);
                           everything else ("must not ...", "must avoid ...", "must remain ...")
                           becomes a conduct rule
  process_duties        -> conduct rules (no code can verify "browse only" for a document run)
  soft_preferences      -> soft_prefs (shown to the worker, not enforced)
  blockers              -> blockers (watched by the judge)
  answer_schema (text)  -> answer_schema {"format": text}
  budget_policy         -> runtime_budget_caps become max_* limits (the server still caps them);
                           structural_stall_parameters and lease_action_caps are applied to the
                           controller; other keys are recorded only
  provenance (object)   -> a provenance string that says the source is *claimed* by the input

Conduct rules are shown to the worker and the judge and reported as NOT machine-checked.
Unknown fields, wrong types or an unknown schema value are rejected with a message.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List

from .contracts import (EvidenceObligation, TaskContract, _ID, _kind,
                        validate_controller_settings)

SCHEMA = "cglc-contract-v1"
V1_KEYS = {"contract_schema", "contract_id", "revision_id", "goal", "process_duties",
           "hard_task_constraints", "evidence_obligations", "soft_preferences", "blockers",
           "answer_schema", "budget_policy", "provenance"}
BUDGET_KEYS = {"policy_version", "lease_action_caps", "checkpoint_triggers",
               "structural_stall_parameters", "runtime_budget_caps",
               "requires_runtime_configuration", "configuration_note"}
RUNTIME_CAPS = {"worker_actions": "max_tool_calls", "controller_calls": None,
                "total_tokens": "max_tokens", "runtime_seconds": "max_seconds"}

# "must show/be/address/use/distinguish ..." constraints are provable by a quote
_CONTENT = re.compile(r"^must (show|be|address|use|distinguish)\b", re.I)
_CONDITIONAL = re.compile(r"\b(when relevant|if visible|if relevant|where relevant)\b", re.I)


def from_cglc_v1(d: Dict[str, Any]) -> TaskContract:
    if d.get("contract_schema") != SCHEMA:
        raise ValueError(f"unknown contract_schema {d.get('contract_schema')!r}; this server reads "
                         f"{SCHEMA!r} or the native format (no contract_schema field)")
    unknown = sorted(set(d) - V1_KEYS)
    if unknown:
        raise ValueError(f"unknown {SCHEMA} field(s) {unknown}; allowed: {sorted(V1_KEYS)}")

    def text(v: Any, name: str, limit: int) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{name} must be a non-empty string, but got {_kind(v)}")
        if len(v) > limit:
            raise ValueError(f"{name} is too long (max {limit} characters)")
        return v.strip()

    def strings(key: str, limit: int, each: int, required: bool = False) -> List[str]:
        v = d.get(key, [])
        if not isinstance(v, list):
            raise ValueError(f"{key} must be a list of strings, but got {_kind(v)}")
        if len(v) > limit:
            raise ValueError(f"{key} has too many entries (max {limit})")
        out = [text(x, f"{key}[{i}]", each) for i, x in enumerate(v)]
        if required and not out:
            raise ValueError(f"{key} must not be empty")
        return out

    def small_obj(key: str) -> Dict[str, Any]:
        v = d.get(key, {})
        if not isinstance(v, dict):
            raise ValueError(f"{key} must be a JSON object, but got {_kind(v)}")
        try:
            size = len(json.dumps(v, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must contain only plain JSON values (no NaN or Infinity)")
        if size > 4000:
            raise ValueError(f"{key} is too large (max 4000 characters of JSON)")
        return v

    goal = text(d.get("goal"), "goal", 3000)
    duties = strings("process_duties", 10, 300)
    constraints = strings("hard_task_constraints", 12, 300)
    evid = strings("evidence_obligations", 8, 300)
    blockers = strings("blockers", 10, 300)
    ans = d.get("answer_schema", "")
    if ans != "" and (not isinstance(ans, str) or len(ans) > 600):
        raise ValueError(f"answer_schema must be text of at most 600 characters, but got {_kind(ans)}")
    soft = small_obj("soft_preferences")

    # ---- map constraints / obligations ---------------------------------------------------
    obligations: List[EvidenceObligation] = []

    def add(proposition: str, conditional: bool) -> None:
        obligations.append(EvidenceObligation(
            obligation_id=f"ev-{len(obligations)}", proposition=proposition[:400],
            conditional=conditional))

    for t in evid:
        add(f"Quoted evidence for: {t}", bool(_CONDITIONAL.search(t)))
    conduct: List[str] = list(duties)
    for t in constraints:
        if _CONTENT.match(t):
            add(f"The documents show, in a quote, that the answer satisfies: {t}",
                bool(_CONDITIONAL.search(t)))
        else:
            conduct.append(t)
    if len(obligations) > 8:
        raise ValueError("this contract maps to more than 8 evidence obligations (the limit)")
    if not obligations:
        raise ValueError("the contract has no evidence obligations or provable constraints")

    # ---- budget_policy -----------------------------------------------------------------------
    bp = small_obj("budget_policy")
    extra = sorted(set(bp) - BUDGET_KEYS)
    if extra:
        raise ValueError(f"unknown budget_policy field(s) {extra}; allowed: {sorted(BUDGET_KEYS)}")
    native: Dict[str, Any] = {}
    caps = bp.get("runtime_budget_caps", {})
    if not isinstance(caps, dict) or set(caps) - set(RUNTIME_CAPS):
        raise ValueError(f"budget_policy.runtime_budget_caps must be an object with keys from "
                         f"{sorted(RUNTIME_CAPS)}")
    for k, v in caps.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"budget_policy.runtime_budget_caps.{k} must be a positive, finite number")
        if RUNTIME_CAPS[k]:
            native[RUNTIME_CAPS[k]] = v
    for k in ("lease_action_caps", "structural_stall_parameters"):
        if k in bp:
            native[k] = bp[k]
    for k in ("policy_version", "configuration_note"):
        if k in bp:
            native[k] = text(bp[k], f"budget_policy.{k}", 400)
    validate_controller_settings(native)

    # ---- provenance: a claim made by the input, labelled as such ---------------------------------------
    prov = small_obj("provenance")
    claimed = ", ".join(f"{k}={prov[k]}" for k in ("dataset", "task_id") if isinstance(prov.get(k), str))
    rid = d.get("revision_id")
    provenance = (f"user-supplied ({SCHEMA} import; source claimed by the input"
                  + (f": {claimed}" if claimed else "")
                  + (f"; revision_id {str(rid)[:24]}, not verified" if isinstance(rid, str) else "") + ")")

    cid = d.get("contract_id")
    if cid is not None and (not isinstance(cid, str) or not _ID.match(cid)):
        raise ValueError("contract_id may use only letters, digits, '_', '-' and '.' (max 40)")

    c = TaskContract.create(
        goal, evidence_obligations=[], soft_prefs=soft, blockers=blockers,
        answer_schema={"format": ans} if ans else {}, budget_policy=native,
        provenance=provenance, conduct_rules=conduct)
    c.evidence_obligations = obligations
    if cid:
        c.contract_id = cid
    problems = c.validate()
    if problems:
        raise ValueError("; ".join(problems))
    return c
