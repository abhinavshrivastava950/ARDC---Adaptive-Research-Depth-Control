"""Bring-your-own-key LLM client for the CGLC worker and checkpoint judge.

The controller never hardcodes a credential. The key is resolved by the
Anthropic SDK from the caller's environment (``ANTHROPIC_API_KEY``, or an
``ant auth login`` profile), or passed explicitly as ``api_key=``. It is
never stored on the object, logged, or included in ``repr``.

Both LLM roles (worker, checkpoint judge) go through ``LLMClient`` so the
control loop stays provider-neutral and tests can inject a scripted fake.

Notes on the current Claude models (``claude-opus-5-5`` default):
 * No ``temperature`` / ``top_p`` -- sampling params are removed, so the
   BATS ``T_gen`` / ``T_select`` config values are recorded but not sent.
 * Thinking is always on; depth is controlled with ``output_config.effort``.
 * No forced ``tool_choice``; JSON replies use ``output_config.format``.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

DEFAULT_MODEL = "claude-opus-5-5"
MODEL_ENV = "CGLC_MODEL"


class LLMError(RuntimeError):
    """Transient or per-call failure; the control loop may continue."""


class LLMConfigError(LLMError):
    """Missing/invalid credentials or model. Never retried: fail loudly."""


class LLMRefusal(LLMError):
    """The model declined (stop_reason == 'refusal')."""


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def work_tokens(self) -> float:
        """Tokens charged to the CGLC token budget (Sec 14.4).

        Cache reads are excluded: they re-read an already-paid prefix and
        would otherwise make a cached corpus look like fresh work each call.
        """
        return float(self.input_tokens + self.output_tokens
                     + self.cache_creation_input_tokens)


@dataclass
class LLMReply:
    data: Dict[str, Any]
    usage: LLMUsage = field(default_factory=LLMUsage)
    seconds: float = 0.0
    request_id: str = ""


class LLMClient(Protocol):
    model: str

    def complete_json(self, system: List[Dict[str, Any]] | str, user: str,
                      schema: Dict[str, Any], max_tokens: int = 8000,
                      temperature: Optional[float] = None) -> LLMReply:
        ...


def system_blocks(*parts: str, cache_last_stable: Optional[int] = None) -> List[Dict[str, Any]]:
    """Build a system prompt. ``cache_last_stable`` marks that part index
    (e.g. the document corpus) as a cache breakpoint."""
    blocks: List[Dict[str, Any]] = []
    for i, p in enumerate(parts):
        b: Dict[str, Any] = {"type": "text", "text": p}
        if cache_last_stable is not None and i == cache_last_stable:
            b["cache_control"] = {"type": "ephemeral"}
        blocks.append(b)
    return blocks


class AnthropicClient:
    """Claude via the official SDK, BYOK.

    ``client`` lets tests (or callers with a pre-built SDK client, e.g.
    Bedrock/Vertex) inject their own object exposing ``messages.create``.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 effort: str = "medium", fallbacks: bool = False,
                 client: Any = None) -> None:
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self.effort = effort
        self.fallbacks = fallbacks
        self._anthropic = None
        if client is not None:
            self._client = client
            try:
                import anthropic
                self._anthropic = anthropic
            except ImportError:
                pass
            return
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on env
            raise LLMConfigError(
                "The 'anthropic' package is required for LLM mode: "
                "pip install 'cglc-worker[llm]'") from e
        self._anthropic = anthropic
        try:
            # api_key=None -> SDK resolves ANTHROPIC_API_KEY / auth profile.
            self._client = anthropic.Anthropic(api_key=clean_api_key(api_key))
        except Exception as e:
            raise LLMConfigError(_BYOK_HINT) from e

    def __repr__(self) -> str:  # never expose credentials
        return f"AnthropicClient(model={self.model!r}, effort={self.effort!r})"

    def complete_json(self, system, user: str, schema: Dict[str, Any],
                      max_tokens: int = 8000,
                      temperature: Optional[float] = None) -> LLMReply:
        # `temperature` is accepted for interface parity and ignored: current
        # Claude models reject sampling parameters.
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        api = self._client.messages
        if self.fallbacks:
            api = self._client.beta.messages
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        t0 = time.time()
        try:
            resp = api.create(**kwargs)
        except Exception as e:
            raise self._translate(e) from e
        seconds = time.time() - t0

        if getattr(resp, "stop_reason", None) == "refusal":
            det = getattr(resp, "stop_details", None)
            cat = getattr(det, "category", None)
            raise LLMRefusal(f"model declined the request (category={cat})")
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise LLMError("reply truncated at max_tokens; raise max_tokens")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMError("reply contained no text block")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"reply was not valid JSON: {e}") from e
        u = getattr(resp, "usage", None)
        usage = LLMUsage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_creation_input_tokens=int(
                getattr(u, "cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(
                getattr(u, "cache_read_input_tokens", 0) or 0),
        )
        return LLMReply(data=data, usage=usage, seconds=seconds,
                        request_id=str(getattr(resp, "_request_id", "") or ""))

    def _translate(self, e: Exception) -> LLMError:
        # No credential at all: the SDK raises a bare TypeError at request time.
        if isinstance(e, TypeError) and "authentication method" in str(e):
            return LLMConfigError("No credentials found. " + _BYOK_HINT)
        a = getattr(self, "_anthropic", None)
        if a is not None:
            if isinstance(e, a.AuthenticationError):
                return LLMConfigError("Anthropic rejected the API key. " + _BYOK_HINT)
            if isinstance(e, a.PermissionDeniedError):
                return LLMConfigError("API key lacks permission for this model/feature.")
            if isinstance(e, a.NotFoundError):
                return LLMConfigError(f"Unknown model {self.model!r}.")
            if isinstance(e, a.BadRequestError):
                return LLMConfigError(f"Bad request: {getattr(e, 'message', e)}")
            if isinstance(e, a.RateLimitError):
                return LLMError("rate limited (SDK retries exhausted)")
            if isinstance(e, a.APIConnectionError):
                return LLMError("network error talking to the Anthropic API")
            if isinstance(e, a.APIStatusError):
                return LLMError(f"API error {e.status_code}: {getattr(e, 'message', e)}")
        return LLMError(f"{type(e).__name__}: {e}")


# whitespace plus zero-width / bidi-control / BOM characters that copy-paste drags in
_HIDDEN = re.compile("[" + r"\s" + "".join(map(chr, (*range(0x200B, 0x2010), *range(0x202A, 0x202F), 0x2060, 0xFEFF))) + "]")


def clean_api_key(raw: Optional[str]) -> Optional[str]:
    """Normalize a pasted API key.

    Copy-paste often drags in invisible characters (zero-width spaces, BOMs, a
    trailing newline). Those are removed. If anything non-ASCII remains (curly
    dashes, a masked key made of bullet characters), fail with a clear message
    instead of letting the HTTP layer raise a cryptic UnicodeEncodeError.
    """
    if raw is None:
        return None
    k = _HIDDEN.sub("", str(raw))
    if not k:
        return None
    if not k.isascii():
        raise LLMConfigError(
            "Your API key contains a character that is not plain ASCII (often from copying a "
            "masked or formatted key). Copy it again from the provider's console and paste it "
            "with nothing before or after it.")
    return k


def extract_json(text: str) -> Dict[str, Any]:
    """Pull one JSON object out of a model reply.

    Tolerates reasoning tags (<think>...</think>), markdown fences and
    leading/trailing prose, which open-weight models sometimes emit even in
    JSON mode. Raises LLMError if no object can be parsed.
    """
    import re
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        v = json.loads(t)
    except json.JSONDecodeError:
        i = t.find("{")
        if i < 0:
            raise LLMError("reply contained no JSON object")
        try:
            v, _ = json.JSONDecoder().raw_decode(t[i:])
        except json.JSONDecodeError as e:
            raise LLMError(f"reply was not valid JSON: {e}") from e
    if not isinstance(v, dict):
        raise LLMError("reply JSON was not an object")
    return v


def validate_schema(data: Any, schema: Dict[str, Any], path: str = "$",
                    limit: int = 8) -> List[str]:
    """Minimal JSON-Schema check (type/required/enum/items) for providers
    that cannot guarantee the schema. Returns human-readable errors."""
    errs: List[str] = []
    if "enum" in schema and data not in schema["enum"]:
        return [f"{path}: must be one of {schema['enum']}"]
    t = schema.get("type")
    if t == "object":
        if not isinstance(data, dict):
            return [f"{path}: expected an object"]
        for k in schema.get("required", []):
            if k not in data:
                errs.append(f"{path}.{k}: missing")
        for k, sub in schema.get("properties", {}).items():
            if k in data:
                errs += validate_schema(data[k], sub, f"{path}.{k}", limit)
    elif t == "array":
        if not isinstance(data, list):
            return [f"{path}: expected an array"]
        for i, item in enumerate(data):
            errs += validate_schema(item, schema.get("items", {}), f"{path}[{i}]", limit)
    elif t == "string" and not isinstance(data, str):
        errs.append(f"{path}: expected a string")
    elif t == "boolean" and not isinstance(data, bool):
        errs.append(f"{path}: expected true/false")
    elif t == "number" and (isinstance(data, bool) or not isinstance(data, (int, float))):
        errs.append(f"{path}: expected a number")
    return errs[:limit]


def make_llm(provider: str, api_key: Optional[str] = None,
             model: Optional[str] = None, **kw: Any) -> LLMClient:
    """Provider factory: 'anthropic' (Claude) or 'groq' (any Groq model)."""
    p = provider.lower()
    if p in ("anthropic", "claude"):
        return AnthropicClient(model=model, api_key=api_key, **kw)
    if p == "groq":
        from .llm_groq import GroqClient
        return GroqClient(model=model, api_key=api_key, **kw)
    raise LLMConfigError(f"unknown provider {provider!r} (use 'groq' or 'anthropic')")


_BYOK_HINT = (
    "Bring your own key: set the ANTHROPIC_API_KEY environment variable "
    "(or run `ant auth login`). Keys are never read from config files or chat."
)
