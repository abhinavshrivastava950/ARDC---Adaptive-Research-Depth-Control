"""Benchmark harness tests: loaders, contracts, metrics, arms, runner CLI, report."""
import json
from pathlib import Path

import pytest

from cglc.bench import ARMS, BASELINES, REGIMES, load, make_contract, run_item
from cglc.bench import metrics as M
from cglc.bench import report, run as bench_run
from cglc.llm import LLMReply, LLMUsage
from cglc.runner import Runner

FIX = Path(__file__).parent / "fixtures"


# --- loaders / contracts -------------------------------------------------------
def test_hotpot_loader_sanitizes_titles_and_marks_gold():
    items = load("hotpotqa", FIX / "hotpot_mini.json")
    it = next(i for i in items if i.item_id == "hp1")
    assert set(it.gold_titles) == {"Eiffel Tower", "Paris"}
    assert all("#" not in t and '"' not in t for t in it.docs)
    assert it.answers == ["France"] and len(it.docs) == 4


def test_musique_loader_skips_unanswerable_and_uses_aliases():
    items = load("musique", FIX / "musique_mini.jsonl")
    assert [i.item_id for i in items] == ["2hop__1"]
    assert items[0].answers == ["Maria Lopez", "Lopez"] and items[0].qtype == "2hop"
    assert set(items[0].gold_titles) == {"Zed phone", "Nova Corp"}


def test_sampling_is_deterministic_and_max_chars_filters():
    a = load("hotpotqa", FIX / "hotpot_mini.json", n=1, seed=3)
    b = load("hotpotqa", FIX / "hotpot_mini.json", n=1, seed=3)
    assert [i.item_id for i in a] == [i.item_id for i in b]
    assert load("hotpotqa", FIX / "hotpot_mini.json", max_chars=50) == []


def test_contract_regimes():
    it = load("hotpotqa", FIX / "hotpot_mini.json")[0]
    o = make_contract(it, "oracle_docs")
    assert len(o.evidence_obligations) == len(it.gold_titles)
    assert it.answers[0] not in " ".join(x.proposition for x in o.evidence_obligations)  # no answer leak
    g = make_contract(it, "generic")
    assert len(g.evidence_obligations) == 1
    with pytest.raises(ValueError):
        make_contract(it, "nope")


# --- metrics --------------------------------------------------------------------
def test_answer_cleaning_and_scores():
    assert M.clean_answer("Final answer: The Eiffel Tower [Paris].\nextra") == "The Eiffel Tower"
    assert M.exact_match("the eiffel tower!", ["Eiffel Tower"]) == 1.0
    assert M.f1("Eiffel Tower in Paris", ["Eiffel Tower"]) == pytest.approx(0.667, abs=0.01)
    assert M.f1("", ["x"]) == 0.0
    assert M.evidence_recall(["a"], ["a", "b"]) == 0.5


def test_over_execution_counts_steps_after_sufficiency():
    steps = [["a"], ["b"], ["a"], ["b"]]
    assert M.steps_after_sufficient(steps, ["a", "b"]) == 2
    assert M.steps_after_sufficient([["a"]], ["a", "b"]) == 0


def test_derived_flags():
    row = {"em": 0.0, "f1": 0.0, "accepted": True, "evidence_recall": 0.5}
    d = M.derive(row)
    assert d["false_allow"] and d["unsupported_accept"] and not d["false_block"]
    d = M.derive({"em": 1.0, "f1": 1.0, "accepted": False, "evidence_recall": 1.0})
    assert d["false_block"] and not d["false_allow"]


# --- arms ------------------------------------------------------------------------
class OracleFake:
    """Answers HotpotQA-mini correctly, but only cites the first doc on step 1
    and both docs from step 2 on, so controllers have something to do."""
    model = "fake"

    def __init__(self, item):
        self.item, self.worker_calls = item, 0

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        u = LLMUsage(input_tokens=300, output_tokens=60)
        props = schema["properties"]
        if "citations" in props:
            self.worker_calls += 1
            titles = self.item.gold_titles
            use = titles[:1] if self.worker_calls == 1 else titles
            cites = [{"source_id": t, "quote": self.item.docs[t][:30]} for t in use]
            return LLMReply({"query": f"step {self.worker_calls}", "citations": cites,
                             "draft": f"{self.item.answers[0]} [{use[0]}]",
                             "propose_final": True, "contradiction": False, "blocker": ""}, u, 0.0)
        # judge: SUPPORTED only for obligations whose doc has been cited
        cited = {s for s in user.split('"span_id": "')[1:]}
        spans = [x.split('"')[0] for x in cited]
        obls = json.loads(user.split("EVIDENCE OBLIGATIONS: ")[1].split("\n")[0])
        out = []
        for o in obls:
            hit = [s for s in spans if any(t in o["proposition"] and s.startswith(t)
                                           for t in self.item.docs) or "generic" in o["proposition"]]
            ok = bool(hit) if "titled" in o["proposition"] else bool(spans)
            out.append({"obligation_id": o["obligation_id"], "status": "SUPPORTED" if ok else "UNSEEN",
                        "contradicted": False,
                        "receipts": [{"span_id": hit[0] if hit else spans[0], "relation": "supports",
                                      "strength": 0.9, "claim": "c"}] if (ok and spans) else []})
        act = {"progress": "MEDIUM", "verify_weak_claim": 0.0, "move_off_stalled_direction": 0.0,
               "target_open_gap": 1.0, "repeat_risk": 0.0}
        return LLMReply({"obligations": out, "answer_conforms": True, "needs_user": False,
                         "infeasible": False, "blocker": "", "has_alternative": True,
                         "direction": "PRODUCTIVE",
                         "actions": {"CONTINUE": act, "VERIFY": act, "REDIRECT": act},
                         "rationale": "r"}, u, 0.0)


@pytest.fixture
def item():
    return next(i for i in load("hotpotqa", FIX / "hotpot_mini.json") if i.item_id == "hp1")


def test_all_arms_run_and_produce_complete_rows(item):
    for arm in ARMS:
        row = run_item(arm, item, "oracle_docs", OracleFake(item))
        for k in ("answer", "em", "f1", "accepted", "steps", "worker_tokens",
                  "controller_tokens", "evidence_recall", "over_steps", "regime"):
            assert k in row, (arm, k)
        assert row["regime"] == ("none" if arm in BASELINES else "oracle_docs")
        assert row["f1"] == 1.0, arm


def test_fixed_arms_take_exact_step_counts_and_never_use_controller(item):
    s = run_item("fixed_shallow", item, "oracle_docs", OracleFake(item))
    d = run_item("fixed_deep", item, "oracle_docs", OracleFake(item))
    assert (s["steps"], d["steps"]) == (1, 5) and s["controller_tokens"] == d["controller_tokens"] == 0
    assert s["evidence_recall"] == 0.5 and d["evidence_recall"] == 1.0   # shallow under-reads


def test_cglc_withholds_until_oracle_evidence_is_cited(item):
    row = run_item("cglc_full", item, "oracle_docs", OracleFake(item))
    assert row["accepted"] and row["decision"] == "ALLOW_FINALIZE"
    assert row["evidence_recall"] == 1.0 and row["steps"] >= 2     # refused the 1-doc first attempt
    assert row["controller_tokens"] > 0


def test_always_on_judge_checks_after_every_action(item):
    row = run_item("always_on_judge_upper_bound", item, "oracle_docs", OracleFake(item))
    assert row["checkpoints"] == row["steps"]      # lease of 1 -> a checkpoint per action


def test_runner_policy_and_fixed_lease_validation():
    with pytest.raises(ValueError):
        Runner(policy="bogus")
    assert Runner(policy="rules", fixed_lease=3).fixed_lease == 3


# --- CLI + report ------------------------------------------------------------------
def test_cli_dry_run_spends_nothing(capsys):
    rc = bench_run.main(["--dataset", "hotpotqa", "--data", str(FIX / "hotpot_mini.json"), "--n", "2"])
    out = capsys.readouterr().out
    assert rc == 0 and "Dry run only" in out and "LLM calls" in out


def test_report_pairs_items_and_computes_cis(tmp_path):
    rows = []
    for arm, f1s, tok in [("uncontrolled_worker", [0.0, 1.0, 1.0, 0.0], 1000),
                          ("cglc_full", [1.0, 1.0, 1.0, 0.0], 1500)]:
        for i, f in enumerate(f1s):
            rows.append({"dataset": "hotpotqa", "model": "m", "arm": arm, "item_id": f"i{i}",
                         "regime": "none" if arm == "uncontrolled_worker" else "oracle_docs",
                         "f1": f, "em": f, "accepted": True, "decision": "x", "steps": 2,
                         "worker_tokens": tok, "controller_tokens": 0, "elapsed": 1.0,
                         "evidence_recall": 1.0, "over_steps": 0, "draft": "ans"})
    rows.append({**rows[0], "item_id": "only_one_arm"})   # unpaired item must be dropped
    rep = report.build(rows)
    b = rep["blocks"][0]
    assert b["n_items"] == 4 and set(b["arms"]) == {"uncontrolled_worker", "cglc_full"}
    d = b["deltas_vs_ref"]["uncontrolled_worker"]
    assert d["f1"][0] == pytest.approx(0.25) and d["tokens"][0] == pytest.approx(500)
    assert d["f1"][1] <= d["f1"][0] <= d["f1"][2]
    assert "cglc_full" in report.to_markdown(rep)


def test_failed_model_calls_are_flagged_and_excluded_from_paired_report(item):
    from cglc.llm import LLMError

    class FlakyJudge(OracleFake):
        def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
            if "citations" not in schema["properties"]:
                raise LLMError("Groq rate limit hit")
            return super().complete_json(system, user, schema, max_tokens, temperature)

    row = run_item("cglc_full", item, "oracle_docs", FlakyJudge(item), max_checkpoints=3)
    assert not row["accepted"] and row["infra_failures"] >= 1      # fail-closed, and flagged
    ok = run_item("cglc_full", item, "oracle_docs", OracleFake(item))
    assert ok["infra_failures"] == 0 and ok["checkpoint_log"]

    base = {"dataset": "hotpotqa", "model": "m", "item_id": "i0", "f1": 1.0, "em": 1.0,
            "accepted": True, "decision": "x", "steps": 1, "worker_tokens": 10,
            "controller_tokens": 0, "elapsed": 1.0, "evidence_recall": 1.0, "over_steps": 0,
            "draft": "a", "infra_failures": 0}
    rows = [{**base, "arm": "uncontrolled_worker", "regime": "none"},
            {**base, "arm": "cglc_full", "regime": "oracle_docs", "infra_failures": 2}]
    b = report.build(rows)["blocks"][0]
    assert b["n_items"] == 0 and b["failed_items"] == ["i0"]


def test_hf_hotpot_conversion_matches_original_format():
    from cglc.bench.fetch import hf_hotpot_to_original
    hf = [{"id": "x1", "question": "q?", "answer": "a", "type": "bridge", "level": "hard",
           "supporting_facts": {"title": ["T1", "T2"], "sent_id": [0, 1]},
           "context": {"title": ["T1", "T2", "T3"],
                       "sentences": [["s1a.", " s1b."], ["s2."], ["s3."]]}}]
    o = hf_hotpot_to_original(hf)[0]
    assert o["supporting_facts"] == [["T1", 0], ["T2", 1]]
    assert o["context"][0] == ["T1", ["s1a.", " s1b."]] and o["_id"] == "x1"


def test_controller_note_reaches_the_worker_prompt_after_a_failed_checkpoint(item):
    """After VERIFY, the judge's reason is shown to the worker on the next lease."""
    seen = []

    class Spy(OracleFake):
        def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
            if "citations" in schema["properties"]:
                seen.append(user)
            return super().complete_json(system, user, schema, max_tokens, temperature)

    run_item("cglc_full", item, "oracle_docs", Spy(item))
    assert len(seen) >= 2
    assert "CONTROLLER NOTE" not in seen[0]            # first lease: nothing to say yet
    assert "CONTROLLER NOTE" in seen[1]                # second lease carries the judge's reason


def test_judge_failure_text_is_not_forwarded_as_a_note():
    from cglc.worker.llm_worker import LLMDocumentWorker
    w = LLMDocumentWorker({"A": "x" * 20}, make_contract(load("hotpotqa", FIX / "hotpot_mini.json")[0], "generic"),
                          OracleFake(None))
    assert "CONTROLLER NOTE" not in w._task_block("CONTINUE", [], "")
    w.controller_note = "cited span only mentions Santa Fe"
    assert "Santa Fe" in w._task_block("VERIFY", [], "")
