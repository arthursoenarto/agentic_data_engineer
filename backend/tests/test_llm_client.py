from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from backend.llm import LLMClient, LLMUsageTracker


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _StructuredResult(BaseModel):
    value: str


class LLMClientTests(unittest.TestCase):
    def test_anthropic_payload_and_usage_are_normalized(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "model": "claude-sonnet-5",
                    "content": [
                        {"type": "thinking", "thinking": "private"},
                        {"type": "text", "text": "ok"},
                    ],
                    "usage": {"input_tokens": 1000, "output_tokens": 500},
                }
            )

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            tracker = LLMUsageTracker()
            text = LLMClient(
                model="claude-sonnet-5", reasoning_effort="medium"
            ).complete_text(
                "user", system_prompt="system", usage_tracker=tracker
            )

        self.assertEqual(text, "ok")
        self.assertEqual(captured["payload"]["system"], "system")
        self.assertEqual(
            captured["payload"]["output_config"], {"effort": "medium"}
        )
        self.assertAlmostEqual(tracker.summary().estimated_cost_usd, 0.007)

    def test_anthropic_complete_json_uses_native_strict_schema(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "model": "claude-sonnet-5",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": '{"value":"ok"}'}],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                }
            )

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            result = LLMClient(
                model="claude-sonnet-5", reasoning_effort="medium"
            ).complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=_StructuredResult,
            )

        self.assertEqual(result.value, "ok")
        output_config = captured["payload"]["output_config"]
        self.assertEqual(output_config["effort"], "medium")
        self.assertEqual(output_config["format"]["type"], "json_schema")
        self.assertFalse(
            output_config["format"]["schema"]["additionalProperties"]
        )

    def test_gemini_payload_and_usage_are_normalized(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "modelVersion": "gemini-3.7-flash",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"thought": True, "text": "private"},
                                    {"text": "ok"},
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1000,
                        "candidatesTokenCount": 100,
                        "thoughtsTokenCount": 400,
                        "totalTokenCount": 1500,
                    },
                }
            )

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            tracker = LLMUsageTracker()
            text = LLMClient(
                model="gemini-3.7-flash", reasoning_effort="medium"
            ).complete_text(
                "user", system_prompt="system", usage_tracker=tracker
            )

        self.assertEqual(text, "ok")
        self.assertEqual(
            captured["payload"]["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "MEDIUM"},
        )
        usage = tracker.summary()
        self.assertEqual(usage.output_tokens, 500)
        self.assertEqual(usage.reasoning_tokens, 400)
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.002625)

    def test_complete_json_forwards_explicit_model(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output_text": '{"value": "ok"}'})

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            result = LLMClient().complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=_StructuredResult,
                model="gpt-5.5-2026-04-23",
            )

        self.assertEqual(result.value, "ok")
        self.assertEqual(captured["payload"]["model"], "gpt-5.5-2026-04-23")

    def test_instance_reasoning_effort_is_sent_to_responses(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output_text": "ok"})

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            LLMClient(reasoning_effort="xhigh").complete_text("hello")

        self.assertEqual(captured["payload"]["reasoning"], {"effort": "xhigh"})

    def test_complete_json_ignores_trailing_model_commentary(self) -> None:
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch(
                "urllib.request.urlopen",
                return_value=FakeResponse(
                    {"output_text": '{"value": "ok"}\nGenerated the requested files.'}
                ),
            ),
        ):
            result = LLMClient().complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=_StructuredResult,
            )

        self.assertEqual(result.value, "ok")

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

    def test_usage_tracker_records_tokens_and_cost(self) -> None:
        payload = {
            "id": "resp_test",
            "model": "gpt-5.5-2026-04-23",
            "output_text": "ok",
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 400},
                "output_tokens": 500,
                "output_tokens_details": {"reasoning_tokens": 200},
                "total_tokens": 1500,
            },
        }

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("backend.llm.client.load_env"),
            patch("urllib.request.urlopen", return_value=FakeResponse(payload)),
        ):
            tracker = LLMUsageTracker()
            text = LLMClient().complete_text("hello", usage_tracker=tracker)

        usage = tracker.summary()
        self.assertEqual(text, "ok")
        self.assertEqual(usage.call_count, 1)
        self.assertEqual(usage.input_tokens, 1000)
        self.assertEqual(usage.cached_input_tokens, 400)
        self.assertEqual(usage.output_tokens, 500)
        self.assertEqual(usage.reasoning_tokens, 200)
        self.assertEqual(usage.total_tokens, 1500)
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.0182)

    def test_usage_tracker_aggregates_multiple_calls(self) -> None:
        tracker = LLMUsageTracker()
        response = {
            "model": "gpt-5.5",
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 50,
                "output_tokens_details": {"reasoning_tokens": 10},
                "total_tokens": 150,
            },
        }

        tracker.record_response(response, requested_model="gpt-5.5")
        tracker.record_response(response, requested_model="gpt-5.5")

        usage = tracker.summary()
        self.assertEqual(usage.call_count, 2)
        self.assertEqual(usage.input_tokens, 200)
        self.assertEqual(usage.output_tokens, 100)
        self.assertEqual(usage.total_tokens, 300)
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.004)

    def test_usage_tracker_applies_gpt_5_5_long_context_pricing(self) -> None:
        tracker = LLMUsageTracker()
        tracker.record_response(
            {
                "model": "gpt-5.5",
                "usage": {
                    "input_tokens": 300_000,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 10_000,
                    "output_tokens_details": {"reasoning_tokens": 5000},
                    "total_tokens": 310_000,
                },
            },
            requested_model="gpt-5.5",
        )

        self.assertAlmostEqual(tracker.summary().estimated_cost_usd, 3.45)

    def test_usage_tracker_supports_model_sensitivity_pricing(self) -> None:
        expected = {
            "gpt-4o-2024-11-20": 0.0075,
            "gpt-5.1-2025-11-13": 0.00625,
            "gpt-5.2-2025-12-11": 0.00875,
            "gpt-5.4-2026-03-05": 0.0100,
            "gpt-5.5-2026-04-23": 0.0200,
            "gpt-5.6-sol": 0.0140,
            "gpt-6-astra": 0.0350,
            "claude-sonnet-5": 0.0070,
            "claude-opus-5": 0.0175,
            "claude-fable-5-1": 0.0350,
            "gemini-3.7-flash": 0.002625,
            "gemini-3.1-pro-preview": 0.0080,
        }
        for model, expected_cost in expected.items():
            with self.subTest(model=model):
                tracker = LLMUsageTracker()
                tracker.record_response(
                    {
                        "model": model,
                        "usage": {
                            "input_tokens": 1000,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 500,
                            "output_tokens_details": {"reasoning_tokens": 100},
                            "total_tokens": 1500,
                        },
                    },
                    requested_model=model,
                )
                self.assertAlmostEqual(
                    tracker.summary().estimated_cost_usd, expected_cost
                )

    def test_usage_tracker_rejects_unverified_model_pricing(self) -> None:
        tracker = LLMUsageTracker()
        response = {
            "model": "unpriced-model",
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 50,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 150,
            },
        }

        with self.assertRaisesRegex(RuntimeError, "No verified pricing"):
            tracker.record_response(response, requested_model="unpriced-model")


if __name__ == "__main__":
    unittest.main()
