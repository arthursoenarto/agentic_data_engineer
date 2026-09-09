"""Normalized token usage and pricing for supported LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Any

from pydantic import BaseModel, Field


TOKENS_PER_MILLION = Decimal("1000000")


class LLMPricingSnapshot(BaseModel):
    """Rates frozen into an artifact when its LLM cost is calculated."""

    model_family: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    long_context_threshold_tokens: int | None = None
    long_context_input_multiplier: float = 1.0
    long_context_output_multiplier: float = 1.0
    source_url: str
    verified_at: str


class LLMUsageSummary(BaseModel):
    """Aggregate usage for every model call in one agent run."""

    call_count: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    pricing: list[LLMPricingSnapshot]


GPT_5_5_PRICING = LLMPricingSnapshot(
    model_family="gpt-5.5",
    input_usd_per_million=5.00,
    cached_input_usd_per_million=0.50,
    output_usd_per_million=30.00,
    long_context_threshold_tokens=272_000,
    long_context_input_multiplier=2.0,
    long_context_output_multiplier=1.5,
    source_url="https://developers.openai.com/api/docs/models/gpt-5.5",
    verified_at="2026-07-21",
)

GPT_4O_PRICING = LLMPricingSnapshot(
    model_family="gpt-4o",
    input_usd_per_million=2.50,
    cached_input_usd_per_million=1.25,
    output_usd_per_million=10.00,
    source_url="https://developers.openai.com/api/docs/models/gpt-4o",
    verified_at="2026-09-08",
)

GPT_5_1_PRICING = LLMPricingSnapshot(
    model_family="gpt-5.1",
    input_usd_per_million=1.25,
    cached_input_usd_per_million=0.125,
    output_usd_per_million=10.00,
    source_url="https://developers.openai.com/api/docs/models/gpt-5.1",
    verified_at="2026-09-08",
)

GPT_5_2_PRICING = LLMPricingSnapshot(
    model_family="gpt-5.2",
    input_usd_per_million=1.75,
    cached_input_usd_per_million=0.175,
    output_usd_per_million=14.00,
    source_url="https://developers.openai.com/api/docs/models/gpt-5.2",
    verified_at="2026-09-08",
)

GPT_5_4_PRICING = LLMPricingSnapshot(
    model_family="gpt-5.4",
    input_usd_per_million=2.50,
    cached_input_usd_per_million=0.25,
    output_usd_per_million=15.00,
    long_context_threshold_tokens=272_000,
    long_context_input_multiplier=2.0,
    long_context_output_multiplier=1.5,
    source_url="https://developers.openai.com/api/docs/models/gpt-5.4",
    verified_at="2026-09-08",
)

GPT_5_6_SOL_PRICING = LLMPricingSnapshot(
    model_family="gpt-5.6-sol",
    input_usd_per_million=4.00,
    cached_input_usd_per_million=0.40,
    output_usd_per_million=20.00,
    long_context_threshold_tokens=272_000,
    long_context_input_multiplier=2.0,
    long_context_output_multiplier=1.5,
    source_url="https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    verified_at="2026-09-08",
)

GPT_6_ASTRA_PRICING = LLMPricingSnapshot(
    model_family="gpt-6-astra",
    input_usd_per_million=10.00,
    cached_input_usd_per_million=1.00,
    output_usd_per_million=50.00,
    long_context_threshold_tokens=272_000,
    long_context_input_multiplier=2.0,
    long_context_output_multiplier=1.5,
    source_url="https://developers.openai.com/api/docs/models/gpt-6-astra",
    verified_at="2026-09-08",
)

CLAUDE_SONNET_5_PRICING = LLMPricingSnapshot(
    model_family="claude-sonnet-5",
    input_usd_per_million=2.00,
    cached_input_usd_per_million=0.20,
    output_usd_per_million=10.00,
    source_url="https://platform.claude.com/docs/en/about-claude/pricing",
    verified_at="2026-09-08",
)

CLAUDE_OPUS_5_PRICING = LLMPricingSnapshot(
    model_family="claude-opus-5",
    input_usd_per_million=5.00,
    cached_input_usd_per_million=0.50,
    output_usd_per_million=25.00,
    source_url="https://platform.claude.com/docs/en/about-claude/pricing",
    verified_at="2026-09-08",
)

CLAUDE_FABLE_5_1_PRICING = LLMPricingSnapshot(
    model_family="claude-fable-5-1",
    input_usd_per_million=10.00,
    cached_input_usd_per_million=0.25,
    output_usd_per_million=50.00,
    source_url="https://platform.claude.com/docs/en/models/fable-5-1/overview",
    verified_at="2026-09-09",
)

GEMINI_3_7_FLASH_PRICING = LLMPricingSnapshot(
    model_family="gemini-3.7-flash",
    input_usd_per_million=0.75,
    cached_input_usd_per_million=0.075,
    output_usd_per_million=3.75,
    source_url="https://ai.google.dev/gemini-api/docs/pricing",
    verified_at="2026-09-08",
)

GEMINI_3_1_PRO_PRICING = LLMPricingSnapshot(
    model_family="gemini-3.1-pro-preview",
    input_usd_per_million=2.00,
    cached_input_usd_per_million=0.20,
    output_usd_per_million=12.00,
    long_context_threshold_tokens=200_000,
    long_context_input_multiplier=2.0,
    long_context_output_multiplier=1.5,
    source_url="https://ai.google.dev/gemini-api/docs/pricing",
    verified_at="2026-09-08",
)


@dataclass(frozen=True)
class _RecordedUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_usd: Decimal
    pricing: LLMPricingSnapshot


class LLMUsageTracker:
    """Collect normalized provider usage for a single agent run."""

    def __init__(self) -> None:
        self._calls: list[_RecordedUsage] = []

    def record_response(self, response: dict[str, Any], *, requested_model: str) -> None:
        """Record one completed response in the normalized usage shape."""

        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("LLM response did not include token usage; cost cannot be tracked.")

        model = response.get("model") or requested_model
        if not isinstance(model, str):
            raise RuntimeError("LLM response did not identify the billed model.")

        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        input_tokens = _usage_int(usage, "input_tokens")
        cached_input_tokens = _usage_int(input_details, "cached_tokens", default=0)
        output_tokens = _usage_int(usage, "output_tokens")
        reasoning_tokens = _usage_int(output_details, "reasoning_tokens", default=0)
        total_tokens = _usage_int(usage, "total_tokens")

        if cached_input_tokens > input_tokens:
            raise RuntimeError("LLM response reported more cached input tokens than input tokens.")

        pricing = _pricing_for_model(model)
        cost_usd = _calculate_cost(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        )
        self._calls.append(
            _RecordedUsage(
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                pricing=pricing,
            )
        )

    def summary(self) -> LLMUsageSummary:
        """Return aggregate usage and the exact pricing snapshot used."""

        if not self._calls:
            raise RuntimeError("No LLM usage was recorded for this agent run.")

        pricing_by_family = {call.pricing.model_family: call.pricing for call in self._calls}
        total_cost = sum((call.cost_usd for call in self._calls), Decimal("0"))
        return LLMUsageSummary(
            call_count=len(self._calls),
            input_tokens=sum(call.input_tokens for call in self._calls),
            cached_input_tokens=sum(call.cached_input_tokens for call in self._calls),
            output_tokens=sum(call.output_tokens for call in self._calls),
            reasoning_tokens=sum(call.reasoning_tokens for call in self._calls),
            total_tokens=sum(call.total_tokens for call in self._calls),
            estimated_cost_usd=float(total_cost),
            pricing=list(pricing_by_family.values()),
        )


def aggregate_llm_usage(usages: list[LLMUsageSummary]) -> LLMUsageSummary | None:
    """Aggregate summaries from independent calls or trackers."""

    if not usages:
        return None
    pricing = {
        item.model_family: item
        for usage in usages
        for item in usage.pricing
    }
    return LLMUsageSummary(
        call_count=sum(usage.call_count for usage in usages),
        input_tokens=sum(usage.input_tokens for usage in usages),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        reasoning_tokens=sum(usage.reasoning_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        estimated_cost_usd=sum(usage.estimated_cost_usd for usage in usages),
        pricing=list(pricing.values()),
    )


def _usage_int(values: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = values.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"LLM response contained invalid usage field: {key}.")
    return value


def _pricing_for_model(model: str) -> LLMPricingSnapshot:
    if re.fullmatch(r"gpt-4o(?:-\d{4}-\d{2}-\d{2})?", model):
        return GPT_4O_PRICING
    if re.fullmatch(r"gpt-5\.1(?:-\d{4}-\d{2}-\d{2})?", model):
        return GPT_5_1_PRICING
    if re.fullmatch(r"gpt-5\.2(?:-\d{4}-\d{2}-\d{2})?", model):
        return GPT_5_2_PRICING
    if re.fullmatch(r"gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?", model):
        return GPT_5_4_PRICING
    if re.fullmatch(r"gpt-5\.5(?:-\d{4}-\d{2}-\d{2})?", model):
        return GPT_5_5_PRICING
    if model == "gpt-5.6-sol":
        return GPT_5_6_SOL_PRICING
    if model == "gpt-6-astra":
        return GPT_6_ASTRA_PRICING
    if model == "claude-sonnet-5":
        return CLAUDE_SONNET_5_PRICING
    if model == "claude-opus-5":
        return CLAUDE_OPUS_5_PRICING
    if model == "claude-fable-5-1":
        return CLAUDE_FABLE_5_1_PRICING
    if model == "gemini-3.7-flash":
        return GEMINI_3_7_FLASH_PRICING
    if model == "gemini-3.1-pro-preview":
        return GEMINI_3_1_PRO_PRICING
    raise RuntimeError(
        f"No verified pricing is configured for model '{model}'. "
        "Add a pricing snapshot before using this model for tracked pipeline generation."
    )


def _calculate_cost(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    pricing: LLMPricingSnapshot,
) -> Decimal:
    uncached_input_tokens = input_tokens - cached_input_tokens
    input_multiplier = Decimal("1")
    output_multiplier = Decimal("1")
    threshold = pricing.long_context_threshold_tokens
    if threshold is not None and input_tokens > threshold:
        input_multiplier = Decimal(str(pricing.long_context_input_multiplier))
        output_multiplier = Decimal(str(pricing.long_context_output_multiplier))

    input_cost = (
        Decimal(uncached_input_tokens)
        * Decimal(str(pricing.input_usd_per_million))
        * input_multiplier
        / TOKENS_PER_MILLION
    )
    cached_input_cost = (
        Decimal(cached_input_tokens)
        * Decimal(str(pricing.cached_input_usd_per_million))
        * input_multiplier
        / TOKENS_PER_MILLION
    )
    output_cost = (
        Decimal(output_tokens)
        * Decimal(str(pricing.output_usd_per_million))
        * output_multiplier
        / TOKENS_PER_MILLION
    )
    return input_cost + cached_input_cost + output_cost
