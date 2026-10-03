"""Aggregate benchmark JSONL into a results table with paired bootstrap CIs.

    python -m cglc.bench.report results/hotpot.jsonl [more.jsonl ...]

Only items completed by *every* arm in a (regime) block are compared, so
differences are paired and arms are never scored on different item sets.
Items with an infrastructure error (no draft at all) are listed, not hidden.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import metrics as M
from .arms import ARMS

REF = "cglc_full"


def load_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired_ci(a: Sequence[float], b: Sequence[float], reps: int = 2000,
              seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a-b) and its 95% bootstrap CI over items."""
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(seed)
    means = sorted(_mean([d[rnd.randrange(len(d))] for _ in d]) for _ in range(reps))
    return _mean(d), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def _arm_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ders = [M.derive(r) for r in rows]
    acc = [r for r in rows if r["accepted"]]
    tok = [r["worker_tokens"] + r["controller_tokens"] for r in rows]
    return {
        "n": len(rows),
        "f1": _mean([r["f1"] for r in rows]),
        "em": _mean([r["em"] for r in rows]),
        "recall": _mean([r["evidence_recall"] for r in rows]),
        "accept": _mean([float(r["accepted"]) for r in rows]),
        "sel_f1": _mean([r["f1"] for r in acc]) if acc else float("nan"),
        "false_allow": _mean([float(d["false_allow"]) for d in ders]),
        "unsupported": _mean([float(d["unsupported_accept"]) for d in ders]),
        "false_block": _mean([float(d["false_block"]) for d in ders]),
        "tokens": _mean(tok),
        "ctrl_tokens": _mean([r["controller_tokens"] for r in rows]),
        "steps": _mean([r["steps"] for r in rows]),
        "over": _mean([r["over_steps"] for r in rows]),
        "secs": _mean([r["elapsed"] for r in rows]),
    }


def build(rows: List[Dict[str, Any]], ref: str = REF) -> Dict[str, Any]:
    out: Dict[str, Any] = {"blocks": []}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r.get("model", ""))].append(r)
    for (dataset, model), grp in sorted(groups.items()):
        regimes = sorted({r["regime"] for r in grp if r["regime"] != "none"}) or ["none"]
        for regime in regimes:
            by_arm: Dict[str, Dict[str, Dict]] = defaultdict(dict)
            for r in grp:
                if r["regime"] in (regime, "none"):
                    by_arm[r["arm"]][r["item_id"]] = r
            arms = [a for a in ARMS if a in by_arm]
            if not arms:
                continue
            common = set.intersection(*(set(by_arm[a]) for a in arms))
            failed = sorted({i for a in arms for i, r in by_arm[a].items()
                             if i in common and (not r["draft"].strip() or r.get("infra_failures", 0) > 0)})
            usable = sorted(common - set(failed))
            table = {a: _arm_stats([by_arm[a][i] for i in usable]) for a in arms}
            deltas: Dict[str, Any] = {}
            if ref in by_arm and usable:
                def tot(a: str, i: str) -> float:
                    r = by_arm[a][i]
                    return r["worker_tokens"] + r["controller_tokens"]
                for a in arms:
                    if a == ref:
                        continue
                    deltas[a] = {
                        "f1": paired_ci([by_arm[ref][i]["f1"] for i in usable],
                                        [by_arm[a][i]["f1"] for i in usable]),
                        "tokens": paired_ci([tot(ref, i) for i in usable],
                                            [tot(a, i) for i in usable]),
                        "false_allow": paired_ci(
                            [float(M.derive(by_arm[ref][i])["false_allow"]) for i in usable],
                            [float(M.derive(by_arm[a][i])["false_allow"]) for i in usable]),
                    }
            out["blocks"].append({"dataset": dataset, "model": model, "regime": regime,
                                  "n_items": len(usable), "failed_items": failed,
                                  "arms": table, "deltas_vs_ref": deltas, "ref": ref})
    return out


def _f(x: float, pct: bool = False, d: int = 3) -> str:
    if x != x:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.{d}f}"


def _ci(t: Sequence[float], k: int = 3) -> str:
    return f"{t[0]:+.{k}f} [{t[1]:+.{k}f}, {t[2]:+.{k}f}]"


def to_markdown(rep: Dict[str, Any]) -> str:
    L: List[str] = []
    for b in rep["blocks"]:
        L.append(f"### {b['dataset']} / contract regime: `{b['regime']}` / model: `{b['model'] or '?'}` "
                 f"(n = {b['n_items']} paired items)\n")
        if b["failed_items"]:
            L.append(f"_Excluded for infrastructure failures (no draft, or a model call failed, e.g. rate limit): {', '.join(b['failed_items'])}_\n")
        L.append("| arm | F1 | EM | evid. recall | accepted | selective F1 | false-allow | unsupported-accept "
                 "| false-block | tokens | ctrl tok | steps | over-exec | s |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a, s in b["arms"].items():
            L.append(f"| {a} | {_f(s['f1'])} | {_f(s['em'])} | {_f(s['recall'])} | {_f(s['accept'], True)} | "
                     f"{_f(s['sel_f1'])} | {_f(s['false_allow'], True)} | {_f(s['unsupported'], True)} | "
                     f"{_f(s['false_block'], True)} | {s['tokens']:.0f} | {s['ctrl_tokens']:.0f} | "
                     f"{_f(s['steps'], d=2)} | {_f(s['over'], d=2)} | {_f(s['secs'], d=1)} |")
        if b["deltas_vs_ref"]:
            L.append(f"\n**`{b['ref']}` minus each arm** (paired, 95% bootstrap CI; positive F1 = CGLC better, "
                     "negative tokens = CGLC cheaper):\n")
            L.append("| vs arm | dF1 [CI] | dtokens [CI] | dfalse-allow [CI] |")
            L.append("|---|---|---|---|")
            for a, d in b["deltas_vs_ref"].items():
                L.append(f"| {a} | {_ci(d['f1'])} | {_ci(d['tokens'], 0)} | {_ci(d['false_allow'])} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--json", help="also write the aggregate to this file")
    a = ap.parse_args()
    rep = build(load_rows(a.paths), a.ref)
    print(to_markdown(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
