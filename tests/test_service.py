"""Service layer, Groq client, retrieval and RAG tests (no network, no keys)."""
import json

import pytest

from cglc.llm import (LLMConfigError, LLMError, LLMReply, LLMUsage, extract_json,
                      validate_schema)
from cglc.llm_groq import GroqClient, list_groq_models
from cglc.retrieval import BM25Index
from cglc import service

DOC = ("Employees may work remotely up to three days per week. Remote work requires "
       "manager approval. Equipment is provided by the company.")
DOCS = [{"name": "policy.txt", "text": DOC},
        {"name": "hr.md", "text": "Annual leave is 25 days. Parental leave is 16 weeks paid."}]


class SmartFake:
    """Replies by schema shape, so tests don't depend on call order."""
    model = "fake"

    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        self.calls += 1
        props = schema["properties"]
        u = LLMUsage(input_tokens=200, output_tokens=40)
        if set(props) == {"query"}:
            d = {"query": "remote work days manager approval"}
        elif "citations" in props:
            d = {"query": "remote work", "draft": "Up to three days, with manager approval. [policy.txt]",
                 "citations": [{"source_id": "policy.txt",
                                "quote": "work remotely up to three days per week"}],
                 "propose_final": True, "contradiction": False, "blocker": ""}
        else:
            act = {"progress": "MEDIUM", "verify_weak_claim": 0.0,
                   "move_off_stalled_direction": 0.0, "target_open_gap": 1.0, "repeat_risk": 0.0}
            d = {"obligations": [{"obligation_id": "ev-0", "status": "SUPPORTED",
                                  "contradicted": False,
                                  "receipts": [{"span_id": "policy.txt#c0", "relation": "supports",
                                                "strength": 0.9, "claim": "three days"}]}],
                 "answer_conforms": True, "needs_user": False, "infeasible": False,
                 "blocker": "", "has_alternative": False, "direction": "PRODUCTIVE",
                 "actions": {"CONTINUE": act, "VERIFY": act, "REDIRECT": act},
                 "rationale": "ok"}
        return LLMReply(data=d, usage=u, seconds=0.01)


def req(**kw):
    base = {"provider": "groq", "api_key": "gsk_SECRETSECRET", "model": "llama-3.3-70b-versatile",
            "goal": "How many days can staff work remotely?", "documents": DOCS}
    base.update(kw)
    return base


def test_offline_run_needs_no_key_and_reports_structure():
    r = service.run_task({"provider": "offline", "goal": "remote work days approval",
                          "documents": DOCS})
    assert r["ok"] and r["mode"] == "offline" and r["decision"] == "ALLOW_FINALIZE"
    assert r["checkpoints"] and r["steps"] and r["evidence"][0]["status"] == "SUPPORTED"
    assert r["checkpoints"][0]["gates"]["C_evidence"] is True
    assert r["spend"]["controller_tokens"] == 0  # rule judge costs nothing
    assert r["warnings"]


def test_full_mode_with_llm_end_to_end_and_key_not_echoed():
    f = SmartFake()
    r = service.run_task(req(), llm_factory=lambda *a, **k: f)
    assert r["decision"] == "ALLOW_FINALIZE" and r["mode"] == "full"
    ev = r["evidence"][0]["receipts"][0]
    assert ev["source"] == "policy.txt" and "three days" in ev["text"]
    assert "gsk_SECRET" not in json.dumps(r)
    assert r["spend"]["worker"]["tokens"] > 0


def test_rag_mode_retrieves_and_logs_queries():
    f = SmartFake()
    r = service.run_task(req(mode="rag"), llm_factory=lambda *a, **k: f)
    assert r["mode"] == "rag" and r["decision"] == "ALLOW_FINALIZE"
    s = r["steps"][0]
    assert s["query"] == "remote work days manager approval" and s["chunks"]


def test_auto_mode_switches_to_rag_for_large_documents():
    big = {"name": "big.txt", "text": ("Filler sentence about nothing. " * 400) + DOC}
    f = SmartFake()
    r = service.run_task(req(documents=[big]), llm_factory=lambda *a, **k: f)
    assert r["mode"] == "rag"


@pytest.mark.parametrize("bad", [
    {"provider": "nope"}, {"goal": "  "}, {"documents": []},
    {"documents": [{"name": "x", "text": "  "}]}, {"model": "bad model!"},
    {"depth": "huge"}, {"mode": "weird"},
])
def test_input_validation(bad):
    status, body = service.handle_run(json.dumps(req(**bad)).encode())
    assert status == 400 and body["kind"] == "input" and not body["ok"]


def test_handle_run_rejects_non_json_and_scrubs_key_from_errors():
    assert service.handle_run(b"not json")[0] == 400

    def boom(*a, **k):
        raise LLMConfigError("bad key gsk_SECRETSECRET rejected")
    status, body = service.handle_run(json.dumps(req()).encode(), llm_factory=boom)
    assert status == 400 and "gsk_SECRET" not in body["error"] and "***" in body["error"]


def test_unexpected_error_does_not_leak_details():
    def boom(*a, **k):
        raise RuntimeError("internal path /srv/x key gsk_SECRETSECRET")
    status, body = service.handle_run(json.dumps(req()).encode(), llm_factory=boom)
    assert status == 500 and "gsk_" not in body["error"] and "/srv" not in body["error"]


def test_doc_names_are_sanitized_and_deduped():
    docs = service._clean_docs([{"name": "a#b<c>.txt", "text": "x1"},
                                {"name": "a#b<c>.txt", "text": "x2"}])
    assert list(docs) == ["abc.txt", "abc.txt-2"]


def test_time_limit_stops_with_report_blocked_not_a_crash():
    r = service.run_task({"provider": "offline", "goal": "remote work", "documents": DOCS},
                         time_limit=-1)
    assert r["decision"] == "REPORT_BLOCKED"


def test_require_all_docs_is_enforced_as_hard_duty():
    f = SmartFake()  # only ever cites policy.txt
    r = service.run_task(req(require_all_docs=True), llm_factory=lambda *a, **k: f)
    assert r["decision"] != "ALLOW_FINALIZE"
    assert any("process duty" in x for x in r["reasons"])


# --- retrieval -----------------------------------------------------------
def test_bm25_ranks_relevant_chunk_and_excludes_seen():
    idx = BM25Index({"a": "Cats purr softly.\n\nRemote work needs approval from managers.",
                     "b": "Annual leave is generous."})
    top = idx.search("remote work approval", k=2)
    assert top[0].source_id == "a" and "Remote" in top[0].text
    again = idx.search("remote work approval", k=2, exclude=[top[0].chunk_id])
    assert all(c.chunk_id != top[0].chunk_id for c in again)


# --- helpers ---------------------------------------------------------------
def test_extract_json_handles_think_tags_fences_and_prose():
    assert extract_json('<think>hm {x}</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_validate_schema():
    sch = {"type": "object", "required": ["a"], "properties": {
        "a": {"type": "string", "enum": ["x", "y"]}, "n": {"type": "number"}}}
    assert validate_schema({"a": "x", "n": 1.5}, sch) == []
    assert validate_schema({"a": "z"}, sch)
    assert validate_schema({}, sch) == ["$.a: missing"]
    assert validate_schema({"a": "x", "n": True}, sch)


# --- Groq client -------------------------------------------------------------
SCHEMA = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}},
          "additionalProperties": False}


def _resp(content, status=200, usage=(11, 7), finish="stop"):
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}
    return status, {}, json.dumps(body)


def _http(responses, log):
    it = iter(responses)

    def http(method, url, headers, body, timeout):
        log.append((method, url, headers, json.loads(body) if body else None))
        return next(it)
    return http


def test_groq_request_shape_and_usage():
    log = []
    g = GroqClient(model="openai/gpt-oss-20b", api_key="gsk_k1234567",
                   http=_http([_resp('<think>x</think>{"ok": true}')], log))
    out = g.complete_json("sys", "hi", SCHEMA, 500, temperature=0.0)
    m, url, headers, payload = log[0]
    assert url.endswith("/openai/v1/chat/completions") and headers["Authorization"] == "Bearer gsk_k1234567"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_completion_tokens"] == 500 and payload["temperature"] == 0.0
    assert "JSON Schema" in payload["messages"][0]["content"]
    assert out.data == {"ok": True} and out.usage.work_tokens == 18.0


def test_groq_repairs_invalid_json_once_then_fails_closed():
    log = []
    g = GroqClient(api_key="gsk_k1234567", http=_http([_resp("nope"), _resp('{"ok": false}')], log))
    assert g.complete_json("s", "u", SCHEMA).data == {"ok": False}
    assert "invalid" in log[1][3]["messages"][-1]["content"]
    g2 = GroqClient(api_key="gsk_k1234567", http=_http([_resp("nope"), _resp("still nope")], []))
    with pytest.raises(LLMError, match="twice"):
        g2.complete_json("s", "u", SCHEMA)


def test_groq_rate_limit_retries_then_gives_up():
    sleeps = []
    rl = (429, {"retry-after": "3"}, json.dumps({"error": {"message": "tpm"}}))
    g = GroqClient(api_key="gsk_k1234567", sleep=sleeps.append,
                   http=_http([rl, _resp('{"ok": true}')], []))
    assert g.complete_json("s", "u", SCHEMA).data["ok"] is True and sleeps == [3.0]
    g2 = GroqClient(api_key="gsk_k1234567", sleep=lambda s: None, http=_http([rl, rl, rl, rl], []))
    with pytest.raises(LLMError, match="rate limit"):
        g2.complete_json("s", "u", SCHEMA)


def test_groq_auth_and_model_errors_are_config_errors():
    for status in (401, 404):
        g = GroqClient(api_key="gsk_k1234567",
                       http=_http([(status, {}, json.dumps({"error": {"message": "m"}}))], []))
        with pytest.raises(LLMConfigError):
            g.complete_json("s", "u", SCHEMA)


def test_groq_truncation_and_missing_key(monkeypatch):
    g = GroqClient(api_key="gsk_k1234567", http=_http([_resp('{"ok"', finish="length")], []))
    with pytest.raises(LLMError, match="truncated"):
        g.complete_json("s", "u", SCHEMA)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        GroqClient()
    with pytest.raises(LLMConfigError):
        GroqClient(api_key="gsk_k1234567", model="bad model;")


def test_groq_key_not_in_repr_and_models_filtered():
    g = GroqClient(api_key="gsk_TOPSECRET", http=_http([], []))
    assert "TOPSECRET" not in repr(g)
    rows = {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "whisper-large-v3"},
                     {"id": "openai/gpt-oss-20b"}, {"id": "old", "active": False},
                     {"id": "playai-tts"}]}
    http = _http([(200, {}, json.dumps(rows))], [])
    assert list_groq_models("k", http=http) == ["llama-3.3-70b-versatile", "openai/gpt-oss-20b"]


def test_groq_adopts_model_token_cap_from_400_and_retries():
    log = []
    err = (400, {}, json.dumps({"error": {"message": "`max_completion_tokens` must be less than or equal to `4096`, the maximum value for `max_completion_tokens` is less than the `context_window` for this model", "code": ""}}))
    g = GroqClient(api_key="gsk_k1234567", http=_http([err, _resp('{"ok": true}')], log))
    assert g.complete_json("s", "u", SCHEMA, 8000).data["ok"] is True
    assert log[0][3]["max_completion_tokens"] == 8000 and log[1][3]["max_completion_tokens"] == 4096
    # remembered for later calls on the same client
    g._http = _http([_resp('{"ok": true}')], log)
    g.complete_json("s", "u", SCHEMA, 8000)
    assert log[2][3]["max_completion_tokens"] == 4096


def test_parse_duration_formats():
    from cglc.llm_groq import parse_duration
    assert parse_duration("697ms") == pytest.approx(0.697)
    assert parse_duration("6.2s") == pytest.approx(6.2)
    assert parse_duration("1m26.4s") == pytest.approx(86.4)
    assert parse_duration("2h3m50s") == 2 * 3600 + 3 * 60 + 50
    assert parse_duration("") == 0.0


def test_groq_waits_for_token_window_instead_of_hitting_429():
    now = [100.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    def resp(remaining):
        r = _resp('{"ok": true}')
        return r[0], {"x-ratelimit-remaining-tokens": str(remaining),
                      "x-ratelimit-reset-tokens": "6.0s"}, r[2]

    g = GroqClient(api_key="gsk_k1234567", sleep=sleep, clock=lambda: now[0],
                   http=_http([resp(200), resp(7000)], []))
    g.complete_json("s", "u" * 3000, SCHEMA)       # leaves only 200 tokens in the window
    g.complete_json("s", "u" * 3000, SCHEMA)       # needs ~2.5k -> must wait for the reset
    assert len(sleeps) == 1 and 5.5 < sleeps[0] < 7
    # plenty of room -> no waiting
    g2 = GroqClient(api_key="gsk_k1234567", sleep=sleep, clock=lambda: now[0],
                    http=_http([resp(7900), resp(7800)], []))
    g2.complete_json("s", "u", SCHEMA); g2.complete_json("s", "u", SCHEMA)
    assert len(sleeps) == 1


def test_groq_pacing_waits_only_for_the_shortfall_not_the_full_reset():
    now = [100.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    def resp(rem):
        r = _resp('{"ok": true}')
        return r[0], {"x-ratelimit-remaining-tokens": str(rem), "x-ratelimit-limit-tokens": "6000",
                      "x-ratelimit-reset-tokens": "50s"}, r[2]

    g = GroqClient(api_key="gsk_k1234567", sleep=sleep, clock=lambda: now[0],
                   http=_http([resp(1000), resp(5000)], []))
    g.complete_json("s", "u" * 3000, SCHEMA)      # 1000 left, refill 100 tokens/s
    g.complete_json("s", "u" * 3000, SCHEMA)      # needs ~2500 -> ~15 s, far less than the 50 s reset
    assert len(sleeps) == 1 and 10 < sleeps[0] < 20


def test_groq_fails_fast_when_the_wait_would_overrun_the_deadline():
    now = [100.0]

    def resp(rem):
        r = _resp('{"ok": true}')
        return r[0], {"x-ratelimit-remaining-tokens": str(rem), "x-ratelimit-limit-tokens": "6000",
                      "x-ratelimit-reset-tokens": "50s"}, r[2]

    g = GroqClient(api_key="gsk_k1234567", sleep=lambda s: now.__setitem__(0, now[0] + s),
                   clock=lambda: now[0], http=_http([resp(100), resp(5000)], []))
    g.deadline = 105.0                              # only 5 s left in this run
    g.complete_json("s", "u" * 3000, SCHEMA)
    with pytest.raises(LLMError, match="out of time"):
        g.complete_json("s", "u" * 3000, SCHEMA)
    assert g.deadline_hit is True


def test_reasoning_effort_low_only_for_gpt_oss_models():
    log = []
    GroqClient(model="openai/gpt-oss-120b", api_key="gsk_k1234567",
               http=_http([_resp('{"ok": true}')], log)).complete_json("s", "u", SCHEMA)
    GroqClient(model="qwen/qwen3.8-27b", api_key="gsk_k1234567",
               http=_http([_resp('{"ok": true}')], log)).complete_json("s", "u", SCHEMA)
    assert log[0][3]["reasoning_effort"] == "low" and "reasoning_effort" not in log[1][3]


def test_time_limit_is_reported_as_such_and_not_as_missing_evidence():
    r = service.run_task({"provider": "offline", "goal": "remote work", "documents": DOCS},
                         time_limit=-1)
    assert r["decision"] == "REPORT_BLOCKED" and r["time_limit_hit"] is True
    assert "time limit" in r["blocked_condition"] and "not proof" in r["blocked_condition"]
    ok = service.run_task({"provider": "offline", "goal": "remote work days approval",
                           "documents": DOCS})
    assert ok["time_limit_hit"] is False


def test_rag_reads_each_documents_opening_on_the_first_step_and_overviews_work():
    from cglc.retrieval import BM25Index
    idx = BM25Index({"big": "Overview: Acme is a payments platform.\n\n" + "\n\n".join(
        f"Section {i}: Acme acme acme detail {i}." for i in range(30))})
    assert idx.openings(per_doc=1)[0].chunk_id == "big#c0"
