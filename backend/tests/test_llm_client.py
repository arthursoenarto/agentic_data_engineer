from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from backend.llm.client import LLMClient


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LLMClientTests(unittest.TestCase):
    def test_web_search_payload_is_explicit(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output_text": "ok"})

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            text = LLMClient().complete_text("hello", web_search=True, web_search_required=True)

        self.assertEqual(text, "ok")
        self.assertEqual(captured["payload"]["tools"], [{"type": "web_search"}])
        self.assertEqual(captured["payload"]["tool_choice"], "required")

    def test_plain_payload_has_no_tools(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output_text": "ok"})

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            LLMClient().complete_text("hello")

        self.assertNotIn("tools", captured["payload"])


if __name__ == "__main__":
    unittest.main()
