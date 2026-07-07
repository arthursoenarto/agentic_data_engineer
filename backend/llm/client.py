"""OpenAI-backed LLM client.

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
from typing import Any, TypeVar

from backend.env import load_env

StructuredResponseT = TypeVar("StructuredResponseT")

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.5"


def _extract_json_text(text: str) -> str:
    """Extract a JSON object/array from an LLM response that may include fences."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    if stripped.startswith(("{", "[")):
        return stripped

    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index != -1]
    if not starts:
        return stripped

    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    return stripped[start : end + 1] if end >= start else stripped


class LLMClient:
    """Small OpenAI Responses API client for backend agents."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        load_env()

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the shell environment.")

    def complete_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        max_output_tokens: int = 128,
        web_search: bool = False,
        web_search_required: bool = False,
        search_context_size: str | None = None,
    ) -> str:
        """Return plain text for a simple prompt."""

        payload: dict[str, Any] = {
            "model": model or self.model,
            "input": prompt
            if not system_prompt
            else [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": max_output_tokens,
        }
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

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"OpenAI API request failed: {error}") from error

        if isinstance(data.get("output_text"), str):
            return data["output_text"]

        text_chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    text_chunks.append(content["text"])
        return "\n".join(text_chunks) if text_chunks else json.dumps(data, indent=2)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredResponseT],
        max_output_tokens: int = 8192,
        web_search: bool = False,
        web_search_required: bool = False,
        search_context_size: str | None = None,
    ) -> StructuredResponseT:
        """Return JSON validated as the requested Pydantic v2 model."""

        schema = response_model.model_json_schema()
        text = self.complete_text(
            "\n".join(
                [
                    user_prompt,
                    "",
                    "Return only JSON matching this schema:",
                    json.dumps(schema, indent=2),
                ]
            ),
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            web_search=web_search,
            web_search_required=web_search_required,
            search_context_size=search_context_size,
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
