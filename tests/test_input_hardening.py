"""Only a valid contract gets through. Everything else is a clean 4xx, never a crash, and the
worker/LLM is never started for it (so a bad input costs no tokens)."""
import json

import pytest

from cglc import service

DOCS = [{"name": "a.txt", "text": "Employees may work remotely up to three days per week."}]
BASE = {"goal": "remote days?", "evidence_obligations": ["a"]}


def body(contract, **over):
    d = {"provider": "groq", "api_key": "gsk_" + "a" * 52, "documents": DOCS, "contract": contract}
    d.update(over)
    return json.dumps(d).encode()


REJECTED_CONTRACTS = {
    "number": 5, "float": 5.5, "true": True, "null": None, "string": "hello", "list": [], "list of numbers": [1, 2],
    "empty object": {}, "unknown field": {**BASE, "extra": 1},
    "goal number": {**BASE, "goal": 5}, "goal list": {**BASE, "goal": ["x"]}, "goal blank": {**BASE, "goal": "  "},
    "no goal": {"evidence_obligations": ["a"]}, "no obligations": {"goal": "g"},
    "obligations number": {**BASE, "evidence_obligations": 5}, "obligations object": {**BASE, "evidence_obligations": {"a": 1}},
    "obligations null": {**BASE, "evidence_obligations": None},
    "obligation number": {**BASE, "evidence_obligations": [5]}, "obligation null": {**BASE, "evidence_obligations": [None]},
    "obligation empty object": {**BASE, "evidence_obligations": [{}]}, "obligation nested list": {**BASE, "evidence_obligations": [["x"]]},
    "proposition number": {**BASE, "evidence_obligations": [{"proposition": 5}]},
    "receipts string": {**BASE, "evidence_obligations": [{"proposition": "p", "required_receipts": "2"}]},
    "receipts float": {**BASE, "evidence_obligations": [{"proposition": "p", "required_receipts": 2.0}]},
    "receipts bool": {**BASE, "evidence_obligations": [{"proposition": "p", "required_receipts": True}]},
    "receipts too big": {**BASE, "evidence_obligations": [{"proposition": "p", "required_receipts": 99}]},
    "weight negative": {**BASE, "evidence_obligations": [{"proposition": "p", "weight": -1}]},
    "weight string": {**BASE, "evidence_obligations": [{"proposition": "p", "weight": "1"}]},
    "obligation id bad chars": {**BASE, "evidence_obligations": [{"obligation_id": "a\"b\n<c>#", "proposition": "p"}]},
    "duplicate ids": {**BASE, "evidence_obligations": [{"obligation_id": "x", "proposition": "a"}, {"obligation_id": "x", "proposition": "b"}]},
    "duties object": {**BASE, "process_duties": {"check": "use_every_document"}}, "duties number": {**BASE, "process_duties": 3},
    "duties free text": {**BASE, "process_duties": ["read everything"]}, "duty null": {**BASE, "process_duties": [None]},
    "duty unknown check": {**BASE, "process_duties": [{"check": "vibes"}]},
    "duty n string": {**BASE, "process_duties": [{"check": "min_distinct_sources", "n": "2"}]},
    "duty n too big for the documents": {**BASE, "process_duties": [{"check": "min_distinct_sources", "n": 10 ** 30}]},
    "duty document number": {**BASE, "process_duties": [{"check": "cite_document", "document": 5}]},
    "duty document missing": {**BASE, "process_duties": [{"check": "cite_document", "document": "ghost.pdf"}]},
    "duty id bad chars": {**BASE, "process_duties": [{"check": "use_every_document", "duty_id": "x y\n"}]},
    "blockers string": {**BASE, "blockers": "oops"}, "blockers numbers": {**BASE, "blockers": [1, 2]},
    "blockers null": {**BASE, "blockers": None},
    "soft_prefs list": {**BASE, "soft_prefs": [1]}, "soft_prefs string": {**BASE, "soft_prefs": "x"},
    "soft_prefs null": {**BASE, "soft_prefs": None}, "soft_prefs huge": {**BASE, "soft_prefs": {"k": "x" * 5000}},
    "answer_schema string": {**BASE, "answer_schema": "x"}, "answer_schema null": {**BASE, "answer_schema": None},
    "budget string": {**BASE, "budget_policy": "x"}, "budget negative": {**BASE, "budget_policy": {"max_tokens": -5}},
    "budget string value": {**BASE, "budget_policy": {"max_tokens": "9"}}, "budget bool": {**BASE, "budget_policy": {"max_tokens": True}},
    "revision string": {**BASE, "revision": "1"}, "revision negative": {**BASE, "revision": -1},
    # a typo must not silently weaken a hard obligation (unknown keys inside duties / obligations)
    "obligation typo key": {**BASE, "evidence_obligations": [{"proposition": "a", "required_recepits": 3}]},
    "obligation weight typo": {**BASE, "evidence_obligations": [{"proposition": "a", "wieght": 5}]},
    "duty typo key": {**BASE, "process_duties": [{"check": "use_every_document", "descripton": "x"}]},
    "duty stray param": {**BASE, "process_duties": [{"check": "use_every_document", "n": 2}]},
    "weight above the cap": {**BASE, "evidence_obligations": [{"proposition": "a", "weight": 1e308}]},
    "duplicate propositions": {**BASE, "evidence_obligations": ["same text", "Same   text"]},
    "all obligations conditional": {**BASE, "evidence_obligations": [{"proposition": "a", "conditional": True}]},
    "contract_id bad chars": {**BASE, "contract_id": "a b<c>"}, "contract_id number": {**BASE, "contract_id": 5},
    "clarification string": {**BASE, "clarification_triggers": "ask"},
    "clarification null": {**BASE, "clarification_triggers": None},
    "clarification number entry": {**BASE, "clarification_triggers": [1]},
    "conduct string": {**BASE, "conduct_rules": "be nice"}, "conduct number entry": {**BASE, "conduct_rules": [3]},
    "blocker also a clarification": {**BASE, "blockers": ["login wall"], "clarification_triggers": ["Login wall"]},
}


@pytest.mark.parametrize("label", sorted(REJECTED_CONTRACTS))
def test_malformed_contract_is_a_clean_400_and_the_llm_is_never_started(label):
    started = []

    def factory(*a, **k):
        started.append(1)
        raise AssertionError("the model must not be started for an invalid contract")

    status, out = service.handle_run(body(REJECTED_CONTRACTS[label]), llm_factory=factory)
    assert status == 400 and out["ok"] is False and out["kind"] == "input", (label, out)
    assert "Contract" in out["error"] or "contract" in out["error"] or "ghost" in out["error"] \
        or "distinct" in out["error"], out["error"]
    assert not started
    json.dumps(out, allow_nan=False)                      # strict JSON back


RAW = {
    "number": b"5", "null": b"null", "list": b"[]", "string": b'"x"', "empty": b"", "not json": b"{not json",
    "NaN": b'{"provider":"offline","documents":[{"name":"a","text":"x y z"}],"contract":{"goal":"g","evidence_obligations":["a"],"budget_policy":{"max_tokens":NaN}}}',
    "Infinity": b'{"provider":"offline","documents":[{"name":"a","text":"x y z"}],"contract":{"goal":"g","evidence_obligations":[{"proposition":"a","weight":Infinity}]}}',
    "bad utf8": b"\xff\xfe\x00{",
    "deeply nested array": b"[" * 100000 + b"]" * 100000,
    "deeply nested object": (b'{"a":' * 5000) + b"1" + (b"}" * 5000),
    "deeply nested contract field": b'{"provider":"offline","documents":[{"name":"a","text":"x y z"}],"contract":{"goal":"g","evidence_obligations":["a"],"soft_prefs":' + b"[" * 3000 + b"]" * 3000 + b"}}",
}


@pytest.mark.parametrize("label", sorted(RAW))
def test_malformed_request_bodies_never_crash(label):
    status, out = service.handle_run(RAW[label])
    assert status == 400 and out["ok"] is False and out["kind"] == "input", (label, out)
    json.dumps(out, allow_nan=False)


OTHER = {
    "documents number": {"provider": "offline", "contract": BASE, "documents": 5},
    "documents list of numbers": {"provider": "offline", "contract": BASE, "documents": [1, 2]},
    "document text number": {"provider": "offline", "contract": BASE, "documents": [{"name": "a", "text": 5}]},
    "document name number": {"provider": "offline", "contract": BASE, "documents": [{"name": 5, "text": "x y z"}]},
    "provider number": {"provider": 5, "contract": BASE, "documents": DOCS},
    "model number": {"provider": "groq", "api_key": "k" * 20, "model": 5, "contract": BASE, "documents": DOCS},
    "api_key number": {"provider": "groq", "api_key": 12345678901234567890, "contract": BASE, "documents": DOCS},
    "depth number": {"provider": "offline", "contract": BASE, "documents": DOCS, "depth": 5},
    "simple obligations number": {"provider": "offline", "goal": "g", "obligations": 5, "documents": DOCS},
    "simple obligations object": {"provider": "offline", "goal": "g", "obligations": {"a": 1}, "documents": DOCS},
    "simple obligations numbers": {"provider": "offline", "goal": "g", "obligations": [1], "documents": DOCS},
    "simple goal number": {"provider": "offline", "goal": 5, "documents": DOCS},
    "simple goal list": {"provider": "offline", "goal": ["a"], "documents": DOCS},
    "require_all_docs string": {"provider": "offline", "goal": "g", "require_all_docs": "no", "documents": DOCS},
}


@pytest.mark.parametrize("label", sorted(OTHER))
def test_other_wrongly_typed_request_fields_are_rejected_not_coerced(label):
    status, out = service.handle_run(json.dumps(OTHER[label]).encode())
    assert status == 400 and out["ok"] is False, (label, out)
    json.dumps(out, allow_nan=False)


def test_huge_but_finite_budget_numbers_are_clamped_to_the_server_cap():
    r = service.run_task({"provider": "offline", "documents": DOCS,
                          "contract": {**BASE, "budget_policy": {"max_tokens": 1e308, "max_tool_calls": 10 ** 9}}})
    assert r["spend"]["limits"]["tokens"] == service.HARD_MAX_TOKENS
    assert r["spend"]["limits"]["tool_calls"] == service.HARD_MAX_TOOLS


def test_responses_are_strict_json_even_if_a_non_finite_number_slips_through():
    assert service._finite({"a": float("nan"), "b": [float("inf"), 1.5], "c": {"d": float("-inf")}}) == \
        {"a": None, "b": [None, 1.5], "c": {"d": None}}


def test_the_web_endpoint_returns_json_errors_for_garbage_too():
    pytest.importorskip("flask")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("vercel_app3", Path(__file__).resolve().parents[1] / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = mod.app.test_client()
    for payload in (b"5", b"{not json", b"[" * 100000, b'{"provider":"offline","contract":NaN}'):
        r = c.post("/api/run", data=payload, content_type="application/json")
        assert r.status_code == 400 and r.json["ok"] is False
    r = c.post("/api/run", json={"provider": "offline", "documents": DOCS, "contract": 5})
    assert r.status_code == 400 and "got a number" in r.json["error"]
