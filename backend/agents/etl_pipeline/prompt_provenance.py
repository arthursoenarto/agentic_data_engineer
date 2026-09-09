"""Exact prompt capture shared by pipeline generation treatments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from backend.agents.etl_pipeline.schemas import (
    PromptProvenance,
    PromptSourceProvenance,
    RenderedPromptProvenance,
    stable_json_hash,
)
from backend.llm import LLMClient, LLMUsageTracker, structured_response_prompt


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
StructuredResultT = TypeVar("StructuredResultT")


@dataclass
class PromptRecorder:
    """Collect exact rendered model inputs without changing the LLM boundary."""

    model: str
    reasoning_effort: str | None = None
    source_paths: set[Path] = field(default_factory=set)
    calls: list[RenderedPromptProvenance] = field(default_factory=list)

    def add_source(self, path: Path) -> None:
        self.source_paths.add(path.resolve())

    def record(
        self,
        *,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        max_output_tokens: int,
    ) -> None:
        exact_user_prompt = structured_response_prompt(user_prompt, response_model)
        self.calls.append(
            RenderedPromptProvenance(
                call_name=call_name,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=max_output_tokens,
                system_prompt=system_prompt,
                user_prompt=exact_user_prompt,
                system_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                user_sha256=hashlib.sha256(exact_user_prompt.encode("utf-8")).hexdigest(),
            )
        )

    def provenance(self) -> PromptProvenance:
        sources = [
            PromptSourceProvenance(
                path=path.relative_to(REPOSITORY_ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(self.source_paths)
        ]
        bundle = []
        for call in self.calls:
            item = {
                "call_name": call.call_name,
                "system_sha256": call.system_sha256,
                "user_sha256": call.user_sha256,
                "model": call.model,
                "max_output_tokens": call.max_output_tokens,
                "tools": call.tools,
            }
            if call.reasoning_effort is not None:
                item["reasoning_effort"] = call.reasoning_effort
            bundle.append(item)
        return PromptProvenance(
            sources=sources,
            calls=self.calls,
            rendered_bundle_sha256=stable_json_hash(bundle),
        )


def recorded_complete_json(
    *,
    llm: LLMClient,
    recorder: PromptRecorder,
    call_name: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[StructuredResultT],
    max_output_tokens: int,
    usage_tracker: LLMUsageTracker,
) -> StructuredResultT:
    """Capture an exact structured request, then delegate to the LLM client."""

    recorder.record(
        call_name=call_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=response_model,
        max_output_tokens=max_output_tokens,
    )
    return llm.complete_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=response_model,
        max_output_tokens=max_output_tokens,
        usage_tracker=usage_tracker,
    )
