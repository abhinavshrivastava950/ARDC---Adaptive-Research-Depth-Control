"""Groq client (bring your own key): run any Groq-hosted chat model.

Groq exposes an OpenAI-compatible API. Only ``json_object`` mode is
supported by every model, so the schema is stated in the prompt and the
reply is validated in code, with one repair retry. Standard library only
(no SDK dependency), so it also runs inside a serverless function.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm import (LLMConfigError, LLMError, LLMReply, LLMUsage, extract_json,
                  validate_schema)

GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# Preferred order when picking from a key's available models (UI + tests).
PREFERRED_GROQ_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b",
                         "llama-3.3-70b-versatile"]
GROQ_KEY_ENV = "GROQ_API_KEY"
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,100}$")
_NOT_CHAT = ("whisper", "tts", "guard", "playai", "orpheus", "embed", "transcribe")

_HINT = ("Bring your own key: paste a Groq API key (console.groq.com/keys) or set the "
         "GROQ_API_KEY environment variable. It is never stored.")

# (method, url, headers, body_or_None, timeout) -> (status, headers, text)
HttpFn = Callable[[str, str, Dict[str, str], Optional[bytes], float],
                  Tuple[int, Dict[str, str], str]]


def _urllib_http(method: str, url: str, headers: Dict[str, str],
                 body: Optional[bytes], timeout: float) -> Tuple[int, Dict[str, str], str]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"network error talking to Groq: {getattr(e, 'reason', e)}") from e


_DUR = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration(text: str) -> float:
    """Groq rate-limit reset headers: '697ms', '6.2s', '1m26.4s', '2h3m50s' -> seconds."""
    m = _DUR.match((text or "").strip())
    if not m or not any(m.groups()):
        return 0.0
    h, mi, sec, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + sec + ms / 1000.0


def _flatten(system: Any) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system)


def _err_message(text: str) -> Tuple[str, str]:
    try:
        e = json.loads(text).get("error", {})
        return str(e.get("message", text))[:300], str(e.get("code", ""))
    except Exception:
        return text[:300], ""


class GroqClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: float = 45.0, http: Optional[HttpFn] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 max_rate_retries: int = 3, max_wait: float = 10.0,
                 pace: bool = True, clock: Callable[[], float] = time.time) -> None:
        key = api_key or os.environ.get(GROQ_KEY_ENV)
        if not key:
            raise LLMConfigError(_HINT)
        self.model = model or os.environ.get("CGLC_MODEL") or DEFAULT_GROQ_MODEL
        if not _MODEL_RE.match(self.model):
            raise LLMConfigError(f"invalid model id {self.model!r}")
        self._key = key
        self._http = http or _urllib_http
        self._sleep = sleep
        self.timeout = timeout
        self.max_rate_retries = max_rate_retries
        self.max_wait = max_wait
        self._cap: Optional[int] = None  # model's max_completion_tokens, learned from a 400
        # Proactive pacing from the rate-limit headers of the last response, so
        # long runs wait for the token window to reset instead of hitting 429s.
        self.pace = pace
        self._clock = clock
        self._tokens_left: Optional[float] = None
        self._window_resets_at = 0.0

    def __repr__(self) -> str:  # never expose credentials
        return f"GroqClient(model={self.model!r})"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json", "User-Agent": "cglc-worker"}

    def _note_limits(self, hdrs: Dict[str, str]) -> None:
        rem = hdrs.get("x-ratelimit-remaining-tokens")
        if rem is None:
            return
        try:
            self._tokens_left = float(rem)
        except ValueError:
            return
        self._window_resets_at = self._clock() + parse_duration(hdrs.get("x-ratelimit-reset-tokens", ""))

    def _pace(self, prompt_chars: int) -> None:
        """Sleep until the token window resets if the next call would not fit."""
        if not self.pace or self._tokens_left is None:
            return
        need = prompt_chars / 3.0 + 1500  # prompt estimate + room for reasoning/output
        wait = self._window_resets_at - self._clock()
        if need > self._tokens_left and 0 < wait:
            self._sleep(min(wait + 0.25, 65.0))
            self._tokens_left = None

    def complete_json(self, system, user: str, schema: Dict[str, Any],
                      max_tokens: int = 8000,
                      temperature: Optional[float] = None) -> LLMReply:
        sys_text = (_flatten(system)
                    + "\n\nReturn ONLY one JSON object (no markdown, no commentary) that "
                      "conforms to this JSON Schema:\n"
                    + json.dumps(schema, separators=(",", ":")))
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user},
        ]
        usage = LLMUsage()
        t0 = time.time()
        rate_waits = 0
        repairs = 0
        server_retries = 0
        last_problem = "no attempt made"

        if self._cap:
            max_tokens = min(max_tokens, self._cap)
        while True:
            payload: Dict[str, Any] = {
                "model": self.model, "messages": messages,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            if temperature is not None:
                payload["temperature"] = float(temperature)
            self._pace(len(sys_text) + sum(len(m["content"]) for m in messages[1:]))
            status, hdrs, text = self._http(
                "POST", f"{GROQ_BASE}/chat/completions", self._headers(),
                json.dumps(payload).encode("utf-8"), self.timeout)
            self._note_limits(hdrs)

            if status == 200:
                try:
                    body = json.loads(text)
                    choice = body["choices"][0]
                    content = choice["message"].get("content") or ""
                    u = body.get("usage") or {}
                    usage.input_tokens += int(u.get("prompt_tokens", 0) or 0)
                    usage.output_tokens += int(u.get("completion_tokens", 0) or 0)
                    if choice.get("finish_reason") == "length":
                        raise LLMError("reply truncated at max_tokens; raise max_tokens "
                                       "or use a smaller document")
                    data = extract_json(content)
                    errs = validate_schema(data, schema)
                except LLMError as e:
                    if "truncated" in str(e):
                        raise
                    errs, data, content = [str(e)], None, ""
                except (KeyError, IndexError, ValueError):
                    raise LLMError("unexpected response shape from Groq")
                if not errs:
                    return LLMReply(data=data, usage=usage,
                                    seconds=time.time() - t0,
                                    request_id=hdrs.get("x-request-id", ""))
                last_problem = "; ".join(errs)
                if repairs >= 1:
                    raise LLMError(f"model returned invalid JSON twice: {last_problem}")
                repairs += 1
                messages = messages + [
                    {"role": "assistant", "content": content[:4000] or "{}"},
                    {"role": "user", "content": "That reply was invalid: " + last_problem
                     + ". Reply again with ONLY a valid JSON object matching the schema."}]
                continue

            msg, code = _err_message(text)
            if status in (401, 403):
                raise LLMConfigError("Groq rejected the API key. " + _HINT)
            if status == 404:
                raise LLMConfigError(f"Model {self.model!r} not found for this key. {msg}")
            if status == 400 and code == "json_validate_failed":
                if repairs >= 1:
                    raise LLMError("model could not produce valid JSON; try another model")
                repairs += 1
                continue
            m = re.search(r"max_completion_tokens`?\s+must be less than or equal to\s+`?(\d+)", msg)
            if status == 400 and m and int(m.group(1)) < max_tokens:
                # Small models cap output tokens; adopt the cap and retry.
                max_tokens = self._cap = int(m.group(1))
                continue
            if status == 400 and re.search(r"context|too large|reduce the length", msg, re.I):
                raise LLMError(f"input too large for this model: {msg}")
            if status == 400:
                raise LLMConfigError(f"Groq rejected the request: {msg}")
            if status == 413:
                raise LLMError(f"request too large for this model/plan limit: {msg}")
            if status == 429:
                if rate_waits >= self.max_rate_retries:
                    raise LLMError(f"Groq rate limit hit (try a smaller model/document "
                                   f"or wait a minute): {msg}")
                rate_waits += 1
                try:
                    wait = float(hdrs.get("retry-after", "2"))
                except ValueError:
                    wait = 2.0
                self._sleep(min(max(wait, 0.5), self.max_wait))
                continue
            if status >= 500 and server_retries < 1:
                server_retries += 1
                self._sleep(1.0)
                continue
            raise LLMError(f"Groq API error {status}: {msg}")


def list_groq_models(api_key: str, http: Optional[HttpFn] = None,
                     timeout: float = 20.0) -> List[str]:
    """Chat-capable model ids visible to this key (for the UI dropdown)."""
    if not api_key:
        raise LLMConfigError(_HINT)
    status, _h, text = (http or _urllib_http)(
        "GET", f"{GROQ_BASE}/models",
        {"Authorization": f"Bearer {api_key}", "User-Agent": "cglc-worker"},
        None, timeout)
    if status in (401, 403):
        raise LLMConfigError("Groq rejected the API key. " + _HINT)
    if status != 200:
        raise LLMError(f"Groq models list failed ({status}): {_err_message(text)[0]}")
    try:
        rows = json.loads(text).get("data", [])
    except json.JSONDecodeError:
        raise LLMError("unexpected models response from Groq")
    ids = [r["id"] for r in rows
           if r.get("active", True) and not any(x in r["id"].lower() for x in _NOT_CHAT)]
    return sorted(ids)
