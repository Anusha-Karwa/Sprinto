"""LLM access: JSON-mode calls validated by pydantic, with token + cost tracking.

Decisions:
- temperature 0 everywhere (spec §4): same input → (nearly) same output, so runs are defensible.
- JSON schema is sent to the provider AND the reply is validated by pydantic in our code.
  Invalid JSON → one repair attempt with the validation error shown → still invalid →
  PermanentToolError("invalid_json"). Rate limits / 5xx → TransientToolError. What happens next
  is decided by the retry table (llm.structure → company FAILED, llm.draft → needs_human_draft).
- Thinking is kept at the LOWEST level: these are extraction/drafting tasks, and thinking tokens
  are billed as output (the biggest cost driver).
- Providers are swappable behind LLM_PROVIDER; tests use FakeProvider (no network).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from agent.config import retry_table, settings
from agent.tools.base import PermanentToolError, TransientToolError

T = TypeVar("T", bound=BaseModel)


@dataclass
class RawReply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class Provider(Protocol):
    name: str
    model: str

    def generate(self, system: str, prompt: str, schema: dict) -> RawReply: ...


# ---------------- usage tracking ----------------

@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    by_purpose: dict[str, dict] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        s = settings()
        return round(self.tokens_in / 1e6 * s.price_in_per_m + self.tokens_out / 1e6 * s.price_out_per_m, 6)

    def add(self, purpose: str, tin: int, tout: int) -> None:
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout
        p = self.by_purpose.setdefault(purpose, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        p["calls"] += 1
        p["tokens_in"] += tin
        p["tokens_out"] += tout


def cost_of(tokens_in: int, tokens_out: int) -> float:
    s = settings()
    return round(tokens_in / 1e6 * s.price_in_per_m + tokens_out / 1e6 * s.price_out_per_m, 6)


# ---------------- providers ----------------

class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from google import genai
        s = settings()
        self.model = model or s.gemini_model
        if not (api_key or s.gemini_api_key):
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        self.client = genai.Client(api_key=api_key or s.gemini_api_key)

    def generate(self, system: str, prompt: str, schema: dict) -> RawReply:
        from google.genai import errors, types
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        )
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code in (429, 500, 502, 503, 504):
                raise TransientToolError("rate_limited" if code == 429 else "server_error", str(e)[:200])
            raise PermanentToolError(f"api_error_{code}", str(e)[:300])
        except Exception as e:  # network-level problems
            raise TransientToolError("connection_error", str(e)[:200])
        u = resp.usage_metadata
        tin = (u.prompt_token_count or 0) if u else 0
        # thinking tokens are billed as output → include them so cost reports are honest
        tout = ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0
        return RawReply(text=resp.text or "", tokens_in=tin, tokens_out=tout)


class AnthropicProvider:
    """Optional. Minimal: asks for JSON matching the schema; our pydantic validation is the real check."""
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic
        s = settings()
        self.model = model or s.anthropic_model
        if not (s.anthropic_api_key and self.model):
            raise RuntimeError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL must be set for LLM_PROVIDER=anthropic")
        self.client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        self._anthropic = anthropic

    def generate(self, system: str, prompt: str, schema: dict) -> RawReply:
        a = self._anthropic
        sys = f"{system}\n\nReply with ONLY a JSON object matching this JSON schema:\n{json.dumps(schema)}"
        try:
            msg = self.client.messages.create(model=self.model, max_tokens=4096, temperature=0, system=sys,
                                              messages=[{"role": "user", "content": prompt}])
        except (a.RateLimitError, a.InternalServerError, a.APIConnectionError) as e:
            raise TransientToolError("rate_limited", str(e)[:200])
        except a.APIError as e:
            raise PermanentToolError("api_error", str(e)[:300])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return RawReply(text=text, tokens_in=msg.usage.input_tokens, tokens_out=msg.usage.output_tokens)


class FakeProvider:
    """For tests: returns queued replies (str or Exception) in order and records every prompt."""
    name = "fake"
    model = "fake-model"

    def __init__(self, replies: list[Any]):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str, schema: dict) -> RawReply:
        self.prompts.append((system, prompt))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        text = r if isinstance(r, str) else json.dumps(r)
        return RawReply(text=text, tokens_in=len(prompt) // 4, tokens_out=len(text) // 4)


def make_provider() -> Provider:
    p = settings().llm_provider
    if p == "gemini":
        return GeminiProvider()
    if p == "anthropic":
        return AnthropicProvider()
    raise RuntimeError(f"Unknown LLM_PROVIDER={p!r}")


# ---------------- the one entry point ----------------

class LLM:
    def __init__(self, provider: Provider | None = None, usage: Usage | None = None):
        self._provider = provider
        self.usage = usage or Usage()

    @property
    def provider(self) -> Provider:
        if self._provider is None:
            self._provider = make_provider()
        return self._provider

    def json(self, purpose: str, system: str, prompt: str, schema: type[T]) -> T:
        """Call the model, return a validated `schema` instance.
        Raises TransientToolError (retryable) or PermanentToolError('invalid_json')."""
        max_attempts = retry_table()["caps"]["llm_invalid_json_attempts"]
        json_schema = to_provider_schema(schema)
        last_error = ""
        p = prompt
        for attempt in range(1, max_attempts + 1):
            raw = self.provider.generate(system, p, json_schema)
            self.usage.add(purpose, raw.tokens_in, raw.tokens_out)
            try:
                return schema.model_validate_json(_strip_fences(raw.text))
            except ValidationError as e:
                last_error = str(e)[:800]
                # Repair attempt: show the model exactly what was wrong.
                p = (f"{prompt}\n\n---\nYour previous reply was invalid:\n{last_error}\n"
                     f"Reply again with ONLY valid JSON matching the schema.")
        raise PermanentToolError("invalid_json", f"{purpose}: invalid JSON {max_attempts}x. Last: {last_error}")


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def to_provider_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $refs inlined — some providers handle $defs poorly."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node

    return inline(schema)
