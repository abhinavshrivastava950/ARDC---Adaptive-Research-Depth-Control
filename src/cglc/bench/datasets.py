"""Multi-hop QA dataset adapters: HotpotQA (distractor), 2WikiMultiHopQA, MuSiQue.

Loads the datasets' original local files (no network here) and normalizes
them to ``BenchItem``: a question, gold answers, a small closed corpus of
paragraphs (gold + distractors), and which paragraphs are gold. The corpus
is the item's own context, so every run is a fixed-corpus task (Sec 10.2).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DATASETS = ("hotpotqa", "2wiki", "musique")
_BAD = re.compile(r"[^A-Za-z0-9 ._\-()]")


@dataclass
class BenchItem:
    item_id: str
    dataset: str
    question: str
    answers: List[str]            # gold answer + aliases
    docs: Dict[str, str]          # safe title -> paragraph text
    gold_titles: List[str]        # safe titles of supporting paragraphs
    qtype: str = ""
    meta: Dict[str, str] = field(default_factory=dict)


def safe_title(title: str, taken: Dict[str, str]) -> str:
    """Titles become document ids inside prompts and span ids ('title#c0'),
    so strip '#', quotes and angle brackets; keep them unique."""
    base = _BAD.sub("", title).strip()[:60] or "doc"
    name, n = base, 2
    while name in taken:
        name, n = f"{base}-{n}", n + 1
    return name


def _build(pairs: List[tuple], gold_raw: set) -> tuple[Dict[str, str], List[str]]:
    docs: Dict[str, str] = {}
    gold: List[str] = []
    for raw_title, text in pairs:
        text = text.strip()
        if not text:
            continue
        name = safe_title(raw_title, docs)
        docs[name] = text
        if raw_title in gold_raw and name not in gold:
            gold.append(name)
    return docs, gold


def _hotpot_like(row: dict, dataset: str) -> BenchItem | None:
    """HotpotQA and 2Wiki share: context=[[title,[sent,...]]], supporting_facts=[[title,i]]."""
    pairs = [(t, " ".join(s)) for t, s in row["context"]]
    gold_raw = {t for t, _ in row["supporting_facts"]}
    docs, gold = _build(pairs, gold_raw)
    if not gold:
        return None
    ans = row.get("answer")
    return BenchItem(
        item_id=str(row.get("_id") or row.get("id")), dataset=dataset,
        question=row["question"], answers=[str(ans)] if ans is not None else [],
        docs=docs, gold_titles=gold, qtype=str(row.get("type") or row.get("level") or ""))


def _musique(row: dict) -> BenchItem | None:
    if row.get("answerable") is False:
        return None
    pairs = [(p.get("title", f"p{p['idx']}"), p["paragraph_text"]) for p in row["paragraphs"]]
    # MuSiQue titles can repeat across paragraphs; key gold by position instead.
    docs: Dict[str, str] = {}
    gold: List[str] = []
    for p in row["paragraphs"]:
        name = safe_title(p.get("title") or f"p{p['idx']}", docs)
        docs[name] = p["paragraph_text"].strip()
        if p.get("is_supporting"):
            gold.append(name)
    if not gold:
        return None
    answers = [str(row["answer"])] + [str(a) for a in row.get("answer_aliases", [])]
    hops = len(row.get("question_decomposition", [])) or len(gold)
    return BenchItem(item_id=str(row["id"]), dataset="musique", question=row["question"],
                     answers=answers, docs=docs, gold_titles=gold, qtype=f"{hops}hop")


def _read_rows(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or text.lstrip().startswith("{") and "\n{" in text[:2000]:
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data["data"]


def load(dataset: str, path: str | Path, n: int | None = None, seed: int = 0,
         max_chars: int | None = None) -> List[BenchItem]:
    """Load a deterministic sample of ``n`` items (all if None).

    ``max_chars`` drops items whose corpus is too large to hold in one
    prompt, so a benchmark can stay in whole-document mode.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r} (use one of {DATASETS})")
    conv = _musique if dataset == "musique" else (lambda r: _hotpot_like(r, dataset))
    items = [it for it in (conv(r) for r in _read_rows(Path(path))) if it]
    if max_chars:
        items = [i for i in items if sum(map(len, i.docs.values())) <= max_chars]
    items.sort(key=lambda i: i.item_id)
    if n is not None and n < len(items):
        items = random.Random(seed).sample(items, n)
        items.sort(key=lambda i: i.item_id)
    return items
