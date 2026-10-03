"""Wrap a benchmark item as a CGLC task contract (Sec 4.1, 10.1 step 1).

Two contract regimes are reported separately, never mixed:

* ``oracle_docs``: hard evidence obligations come from the benchmark's gold
  supporting paragraphs (one obligation per gold paragraph: it must be cited
  and relied on). This is the V1 premise "obligations supplied by the
  benchmark". It leaks *which* documents matter, not their content or the
  answer, so it measures the controller under a good contract.
* ``generic``: one content-free obligation (every claim is supported by a
  passage). No benchmark knowledge enters the contract.
"""
from __future__ import annotations

from ..contracts import TaskContract
from ..service import DEFAULT_OBLIGATION
from .datasets import BenchItem

REGIMES = ("oracle_docs", "generic")

ANSWER_FORMAT = (
    "Reply with ONLY the final answer as a short phrase: an entity, name, number, "
    "date, or yes/no, at most eight words, with no explanation. After it you may "
    "add the supporting document names in square brackets, e.g. Paris [France]."
)


def make_contract(item: BenchItem, regime: str) -> TaskContract:
    if regime == "oracle_docs":
        obligations = [f"The document titled “{t}” is cited and relied on in the answer."
                       for t in item.gold_titles]
    elif regime == "generic":
        obligations = [DEFAULT_OBLIGATION]
    else:
        raise ValueError(f"unknown regime {regime!r} (use one of {REGIMES})")
    return TaskContract.create(
        goal=item.question,
        evidence_obligations=obligations,
        answer_schema={"format": ANSWER_FORMAT},
        provenance=f"benchmark:{item.dataset}:{regime}",
    )
