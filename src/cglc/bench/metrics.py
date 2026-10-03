"""Answer and evidence metrics for the benchmark (Sec 11.2).

All definitions are fixed here before any result is produced (see
BENCHMARK.md). ``accepted`` means the arm let the answer through: baselines
accept whatever they produce; CGLC accepts only on ALLOW_FINALIZE.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

_ART = re.compile(r"\b(a|an|the)\b", re.I)
_CITE = re.compile(r"\[[^\]]*\]")
CORRECT_F1 = 0.5  # an answer counts as correct at token-F1 >= 0.5 (or EM)


def clean_answer(draft: str) -> str:
    """Strip citation brackets, 'Final answer:' prefixes, extra lines, trailing dots."""
    t = _CITE.sub("", draft or "").strip()
    t = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", t, flags=re.I)
    t = t.splitlines()[0].strip() if t else ""
    return t.rstrip(" .")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1(pred: str, golds: Sequence[str]) -> float:
    return max((_f1(pred, g) for g in golds), default=0.0)


def evidence_recall(cited: Sequence[str], gold_titles: Sequence[str]) -> float:
    if not gold_titles:
        return 1.0
    return len(set(cited) & set(gold_titles)) / len(set(gold_titles))


def steps_after_sufficient(step_cited: List[Sequence[str]], gold_titles: Sequence[str]) -> int:
    """Over-execution: worker steps taken after cumulative cited evidence first
    covered every gold paragraph. 0 if it never did."""
    seen: set = set()
    for i, c in enumerate(step_cited):
        seen |= set(c)
        if evidence_recall(seen, gold_titles) >= 1.0:
            return len(step_cited) - (i + 1)
    return 0


def derive(row: Dict) -> Dict:
    """Per-item derived flags from a raw result row."""
    correct = row["em"] >= 1.0 or row["f1"] >= CORRECT_F1
    acc = bool(row["accepted"])
    return {
        "correct": correct,
        "false_allow": acc and not correct,                 # approved a wrong answer
        "unsupported_accept": acc and row["evidence_recall"] < 1.0,
        "false_block": (not acc) and correct,               # withheld a correct answer
    }
