"""Small provider-neutral LLM client for OpenAI, Anthropic, and Gemini.

Run a local smoke test:

```bash
python3 backend/llm/client.py
```
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any, Literal, TypeVar

from backend.env import load_env
from backend.llm.usage import LLMUsageTracker

StructuredResponseT = TypeVar("StructuredResponseT")
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
LLMProvider = Literal["openai", "anthropic", "google"]

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_MODEL = "gpt-5.5"


def structured_response_prompt(
    user_prompt: str,
    response_model: type[Any],
) -> str:
    """Render the exact user content sent for a structured response."""

    return "\n".join(
        [
            user_prompt,
            "",
            "Return only JSON matching this schema:",
            json.dumps(response_model.model_json_schema(), indent=2),
        ]
    )


def _anthropic_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the strict object form required by Anthropic structured outputs."""

    transformed = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(transformed)
    return transformed


def _extract_json_text(text: str) -> str:
    """Extract the first complete JSON value from a response with optional framing."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    starts = [
        index
        for index in (stripped.find("{"), stripped.find("["))
        if index != -1
    ]
    if not starts:
        return stripped

    start = min(starts)
    candidate = stripped[start:]
    _, end = json.JSONDecoder().raw_decode(candidate)
    return candidate[:end]


def _provider_for_model(model: str) -> LLMProvider:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"
    return "openai"


def _provider_key_name(provider: LLMProvider) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
    }[provider]


def _provider_api_key(provider: LLMProvider) -> str | None:
    if provider == "google":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return os.environ.get(_provider_key_name(provider))


def _gemini_thinking_level(effort: ReasoningEffort) -> str:
    if effort in {"xhigh", "max"}:
        return "HIGH"
    return effort.upper()


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _normalized_response(
    *,
    model: str,
    text: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "output_text": text,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_input_tokens},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": (
                input_tokens + output_tokens if total_tokens is None else total_tokens
            ),
        },
    }


class LLMClient:
    """Small text/JSON client with normalized usage across supported providers."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        load_env()

        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.provider = _provider_for_model(self.model)
        self.api_key = api_key or _provider_api_key(self.provider)
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            variable = _provider_key_name(self.provider)
            raise RuntimeError(
                f"{variable} is not set. Add it to .env or the shell environment."
            )

    def complete_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int = 128,
        web_search: bool = False,
        web_search_required: bool = False,
        search_context_size: str | None = None,
        usage_tracker: LLMUsageTracker | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return plain text for a simple prompt."""

        effective_model = model or self.model
        provider = _provider_for_model(effective_model)
        if provider != self.provider:
            raise ValueError(
                "A per-call model cannot change provider; create a client for that provider."
            )
        if provider != "openai" and (web_search or web_search_required):
            raise ValueError("Web search is currently supported only by the OpenAI client.")
        effective_reasoning_effort = reasoning_effort or self.reasoning_effort

        if provider == "anthropic":
            text, usage_response = self._complete_anthropic(
                prompt,
                system_prompt=system_prompt,
                model=effective_model,
                reasoning_effort=effective_reasoning_effort,
                max_output_tokens=max_output_tokens,
                response_schema=response_schema,
            )
        elif provider == "google":
            text, usage_response = self._complete_google(
                prompt,
                system_prompt=system_prompt,
                model=effective_model,
                reasoning_effort=effective_reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
        else:
            text, usage_response = self._complete_openai(
                prompt,
                system_prompt=system_prompt,
                model=effective_model,
                reasoning_effort=effective_reasoning_effort,
                max_output_tokens=max_output_tokens,
                web_search=web_search,
                web_search_required=web_search_required,
                search_context_size=search_context_size,
            )

        if usage_tracker is not None:
            usage_tracker.record_response(
                usage_response,
                requested_model=effective_model,
            )
        return text

    def _complete_openai(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        model: str,
        reasoning_effort: ReasoningEffort | None,
        max_output_tokens: int,
        web_search: bool,
        web_search_required: bool,
        search_context_size: str | None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "input": prompt
            if not system_prompt
            else [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if web_search or web_search_required:
            tool: dict[str, Any] = {"type": "web_search"}
            if search_context_size:
                tool["search_context_size"] = search_context_size
            payload["tools"] = [tool]
            if web_search_required:
                payload["tool_choice"] = "required"

        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        data = self._request_json(request, provider_label="OpenAI")

        if isinstance(data.get("output_text"), str):
            return data["output_text"], data

        text_chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    text_chunks.append(content["text"])
        text = "\n".join(text_chunks) if text_chunks else json.dumps(data, indent=2)
        return text, data

    def _complete_anthropic(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        model: str,
        reasoning_effort: ReasoningEffort | None,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        output_config: dict[str, Any] = {}
        if reasoning_effort is not None:
            output_config["effort"] = reasoning_effort
        if response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": _anthropic_json_schema(response_schema),
            }
        if output_config:
            payload["output_config"] = output_config
        request = urllib.request.Request(
            ANTHROPIC_MESSAGES_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        data = self._request_json(request, provider_label="Anthropic")
        text = "\n".join(
            block["text"]
            for block in data.get("content", [])
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        )
        usage = data.get("usage") or {}
        uncached = _nonnegative_int(usage.get("input_tokens"))
        cache_write = _nonnegative_int(usage.get("cache_creation_input_tokens"))
        cache_read = _nonnegative_int(usage.get("cache_read_input_tokens"))
        output = _nonnegative_int(usage.get("output_tokens"))
        normalized = _normalized_response(
            model=str(data.get("model") or model),
            text=text,
            input_tokens=uncached + cache_write + cache_read,
            cached_input_tokens=cache_read,
            output_tokens=output,
            reasoning_tokens=0,
        )
        normalized["provider_metadata"] = {
            "stop_reason": data.get("stop_reason"),
            "stop_sequence": data.get("stop_sequence"),
            "content_block_types": [
                block.get("type")
                for block in data.get("content", [])
                if isinstance(block, dict)
            ],
        }
        return text, normalized

    def _complete_google(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        model: str,
        reasoning_effort: ReasoningEffort | None,
        max_output_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if reasoning_effort is not None:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": _gemini_thinking_level(reasoning_effort)
            }
        request = urllib.request.Request(
            GEMINI_GENERATE_URL.format(model=model),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        data = self._request_json(request, provider_label="Gemini")
        text_chunks: list[str] = []
        for candidate in data.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                if part.get("thought") is not True and isinstance(part.get("text"), str):
                    text_chunks.append(part["text"])
        text = "\n".join(text_chunks)
        usage = data.get("usageMetadata") or {}
        input_tokens = _nonnegative_int(usage.get("promptTokenCount"))
        cached = _nonnegative_int(usage.get("cachedContentTokenCount"))
        visible_output = _nonnegative_int(usage.get("candidatesTokenCount"))
        reasoning = _nonnegative_int(usage.get("thoughtsTokenCount"))
        output_tokens = visible_output + reasoning
        normalized = _normalized_response(
            model=model,
            text=text,
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning,
            total_tokens=_nonnegative_int(
                usage.get("totalTokenCount"),
                default=input_tokens + output_tokens,
            ),
        )
        return text, normalized

    def _request_json(
        self,
        request: urllib.request.Request,
        *,
        provider_label: str,
    ) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{provider_label} API request failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"{provider_label} API request failed: {error}") from error

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredResponseT],
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int = 8192,
        web_search: bool = False,
        web_search_required: bool = False,
        search_context_size: str | None = None,
        usage_tracker: LLMUsageTracker | None = None,
    ) -> StructuredResponseT:
        """Return JSON validated as the requested Pydantic v2 model."""

        text = self.complete_text(
            structured_response_prompt(user_prompt, response_model),
            system_prompt=system_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            web_search=web_search,
            web_search_required=web_search_required,
            search_context_size=search_context_size,
            usage_tracker=usage_tracker,
            response_schema=response_model.model_json_schema(),
        )
        parsed = json.loads(_extract_json_text(text))
        return response_model.model_validate(parsed)  # type: ignore[return-value]


if __name__ == "__main__":
    try:
        print(
            LLMClient().complete_text(
                "Reply with exactly: data discovery client is working",
                max_output_tokens=32,
            )
        )
    except RuntimeError as error:
        print(f"LLM smoke test failed: {error}", file=sys.stderr)
        raise SystemExit(1)
