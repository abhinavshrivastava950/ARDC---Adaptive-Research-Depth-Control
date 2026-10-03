"""Run the Sec 11.1 ladder on a multi-hop QA dataset.

    # 1) see what a run would cost, spending nothing
    python -m cglc.bench.run --dataset hotpotqa --data hotpot_dev_distractor_v1.json \\
        --n 20 --provider groq --model openai/gpt-oss-120b --estimate

    # 2) run it (resumable: re-running skips finished (arm, item) pairs)
    GROQ_API_KEY=... python -m cglc.bench.run --dataset hotpotqa --data ... \\
        --n 20 --provider groq --model openai/gpt-oss-120b --out results/hotpot.jsonl --yes

Every (arm, item) result is appended to the JSONL as soon as it finishes.
Keys are read from the environment only (GROQ_API_KEY / ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ..llm import LLMConfigError, LLMError, make_llm
from . import datasets
from .arms import ARMS, BASELINES, run_item
from .contracts import REGIMES

# rough calls per item, used only for --estimate (measured later from real runs)
_WORKER_CALLS = {"uncontrolled_worker": 1.5, "fixed_shallow": 1, "fixed_deep": 5,
                 "generic_prompt": 1.5, "arch3_fixed_checkpoints": 3, "cglc_no_stall_trigger": 3,
                 "cglc_rule_only": 3, "cglc_full": 3, "always_on_judge_upper_bound": 4}
_JUDGE_CALLS = {"arch3_fixed_checkpoints": 1.5, "cglc_no_stall_trigger": 2, "cglc_rule_only": 2,
                "cglc_full": 2, "always_on_judge_upper_bound": 4}


def estimate(items, arms: List[str], regimes: List[str]) -> Dict[str, float]:
    avg_chars = sum(sum(map(len, i.docs.values())) for i in items) / max(1, len(items))
    worker_tok = avg_chars / 4 + 900          # corpus + prompt + reasoning/output
    judge_tok = 1800.0
    total, calls = 0.0, 0.0
    for a in arms:
        runs = len(items) * (1 if a in BASELINES else len(regimes))
        w, j = _WORKER_CALLS[a], _JUDGE_CALLS.get(a, 0)
        total += runs * (w * worker_tok + j * judge_tok)
        calls += runs * (w + j)
    return {"runs": sum(len(items) * (1 if a in BASELINES else len(regimes)) for a in arms),
            "llm_calls": calls, "tokens": total, "avg_corpus_chars": avg_chars}


def _done(path: Path) -> Set[Tuple[str, str, str, str]]:
    keys: Set[Tuple[str, str, str, str]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("infra_failures", 0) > 0:
                    continue  # a model call failed (e.g. rate limit): run it again
                keys.add((r["arm"], r["item_id"], r["regime"], r.get("model", "")))
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=datasets.DATASETS)
    ap.add_argument("--data", required=True, help="path to the dataset's original local file")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=24000,
                    help="skip items whose corpus exceeds this (keeps whole-document mode)")
    ap.add_argument("--arms", default="all", help="comma list or 'all': " + ",".join(ARMS))
    ap.add_argument("--regime", default="both", choices=list(REGIMES) + ["both"])
    ap.add_argument("--provider", default="groq", choices=["groq", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-model", default=None, help="separate judge model (default: same as worker)")
    ap.add_argument("--worker-temperature", type=float, default=0.0)
    ap.add_argument("--max-checkpoints", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between runs")
    ap.add_argument("--out", default="results/run.jsonl")
    ap.add_argument("--estimate", action="store_true", help="print the cost estimate and exit")
    ap.add_argument("--yes", action="store_true", help="actually spend tokens")
    a = ap.parse_args(argv)

    items = datasets.load(a.dataset, a.data, n=a.n, seed=a.seed, max_chars=a.max_chars)
    arms = ARMS if a.arms == "all" else [x.strip() for x in a.arms.split(",")]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        ap.error(f"unknown arms: {bad}")
    regimes = list(REGIMES) if a.regime == "both" else [a.regime]
    est = estimate(items, arms, regimes)
    print(f"{len(items)} items from {a.dataset}; {len(arms)} arms; regimes={regimes}")
    print(f"~{est['runs']} runs, ~{est['llm_calls']:.0f} LLM calls, ~{est['tokens'] / 1e6:.2f}M tokens "
          f"(corpus avg {est['avg_corpus_chars']:.0f} chars; rough estimate)")
    if a.estimate or not a.yes:
        if not a.yes:
            print("Dry run only. Add --yes to spend tokens.")
        return 0

    # Long benchmark runs should wait out a rate-limit window rather than fail it.
    patience = dict(max_wait=65.0, max_rate_retries=6) if a.provider == "groq" else {}
    llm = make_llm(a.provider, model=a.model, **patience)
    judge_llm = make_llm(a.provider, model=a.judge_model, **patience) if a.judge_model else llm
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    n_done = n_err = 0
    t_start = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for item in items:
            for arm in arms:
                for regime in (["none"] if arm in BASELINES else regimes):
                    key = (arm, item.item_id, regime, llm.model)
                    if key in done:
                        continue
                    try:
                        row = run_item(arm, item, "oracle_docs" if regime == "none" else regime,
                                       llm, judge_llm, a.worker_temperature, 0.0, a.max_checkpoints)
                    except LLMConfigError as e:
                        print(f"\nSTOP: {e}", file=sys.stderr)
                        return 2
                    except LLMError as e:
                        n_err += 1
                        print(f"  ! {arm} {item.item_id}: {e}", file=sys.stderr)
                        continue
                    row["regime"] = regime
                    row["model"] = llm.model
                    row["judge_model"] = judge_llm.model
                    row["worker_temperature"] = a.worker_temperature
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_done += 1
                    print(f"[{n_done}] {item.dataset}/{item.item_id} {arm:<28} {regime:<11} "
                          f"f1={row['f1']:.2f} acc={int(row['accepted'])} steps={row['steps']} "
                          f"tok={row['worker_tokens'] + row['controller_tokens']:.0f} {row['elapsed']}s")
                    if a.sleep:
                        time.sleep(a.sleep)
    print(f"\nfinished {n_done} runs ({n_err} errors) in {time.time() - t_start:.0f}s -> {out}")
    print(f"report: python -m cglc.bench.report {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
