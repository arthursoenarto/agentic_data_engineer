"""LLM proposer for focused, evaluation-guided pipeline source changes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from backend.agents.pipeline_improvement.schemas import (
    ImprovementProposal,
    ImprovementProposalFailureRecord,
    ImprovementProposalRecord,
)
from backend.llm import LLMUsageSummary, LLMUsageTracker


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_CONTEXT_CHARACTERS = 180_000
MAX_FILE_CHARACTERS = 50_000


class StructuredLLM(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> ImprovementProposal: ...


@dataclass(frozen=True)
class ProposalCall:
    """Validated proposal plus the exact rendered context sent to the model."""

    record: ImprovementProposalRecord
    rendered_system_prompt: str
    rendered_user_prompt: str


class ProposalGenerationError(RuntimeError):
    """A failed proposal call with durable prompt and billing provenance."""

    def __init__(
        self,
        *,
        record: ImprovementProposalFailureRecord,
        rendered_system_prompt: str,
        rendered_user_prompt: str,
    ) -> None:
        super().__init__(f"{record.error_type}: {record.error}")
        self.record = record
        self.rendered_system_prompt = rendered_system_prompt
        self.rendered_user_prompt = rendered_user_prompt


class PipelineImprovementAgent:
    """Propose one atomic source patch from F(p), feedback, and search history."""

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def propose(
        self,
        *,
        candidate_dir: Path,
        editable_paths: Sequence[str],
        evaluation_summary: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        max_edits: int = 4,
    ) -> ProposalCall:
        if not 1 <= max_edits <= 4:
            raise ValueError("max_edits must be between one and four.")
        candidate_dir = candidate_dir.resolve()
        source_context, source_sha256 = _source_context(candidate_dir, editable_paths)
        system_prompt = _prompt("default_system.md")
        user_template = _prompt("default_user.md")
        rendered = user_template.format(
            objective_json=json.dumps(evaluation_summary, indent=2, sort_keys=True),
            history_json=json.dumps(list(history), indent=2, sort_keys=True),
            editable_paths_json=json.dumps(sorted(editable_paths), indent=2),
            max_edits=max_edits,
            source_context=source_context,
        )
        if len(rendered) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(
                f"Improvement context exceeds {MAX_CONTEXT_CHARACTERS} characters."
            )
        tracker = LLMUsageTracker()
        try:
            proposal = self._llm.complete_json(
                system_prompt=system_prompt,
                user_prompt=rendered,
                response_model=ImprovementProposal,
                max_output_tokens=12_000,
                usage_tracker=tracker,
            )
            allowed = set(editable_paths)
            disallowed = sorted(
                edit.relative_path
                for edit in proposal.edits
                if edit.relative_path not in allowed
            )
            if disallowed:
                raise ValueError(f"Proposal edits non-editable paths: {disallowed}")
            if len(proposal.edits) > max_edits:
                raise ValueError(f"Proposal contains more than {max_edits} edits.")
            _validate_search_protocol(
                proposal,
                evaluation_summary=evaluation_summary,
                history=history,
            )
        except Exception as error:
            failure = ImprovementProposalFailureRecord(
                model=getattr(self._llm, "model", "unknown"),
                system_prompt_sha256=_sha256(system_prompt),
                user_prompt_sha256=_sha256(rendered),
                parent_source_sha256=source_sha256,
                error_type=type(error).__name__,
                error=str(error),
                llm_usage=_usage_or_none(tracker),
            )
            raise ProposalGenerationError(
                record=failure,
                rendered_system_prompt=system_prompt,
                rendered_user_prompt=rendered,
            ) from error
        record = ImprovementProposalRecord(
            model=getattr(self._llm, "model", "unknown"),
            system_prompt_sha256=_sha256(system_prompt),
            user_prompt_sha256=_sha256(rendered),
            parent_source_sha256=source_sha256,
            proposal=proposal,
            llm_usage=_usage_or_none(tracker),
        )
        return ProposalCall(
            record=record,
            rendered_system_prompt=system_prompt,
            rendered_user_prompt=rendered,
        )


def _source_context(candidate_dir: Path, editable_paths: Sequence[str]) -> tuple[str, str]:
    digest = hashlib.sha256()
    sections: list[str] = []
    for relative in sorted(set(editable_paths)):
        path = (candidate_dir / relative).resolve()
        try:
            path.relative_to(candidate_dir)
        except ValueError as error:
            raise ValueError(f"Editable path escapes candidate: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        content = path.read_text(encoding="utf-8")
        if len(content) > MAX_FILE_CHARACTERS:
            raise ValueError(f"Editable file exceeds context limit: {relative}")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(content.encode("utf-8") + b"\0")
        sections.append(f"FILE: {relative}\n```\n{content}\n```")
    return "\n\n".join(sections), digest.hexdigest()


def _prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _usage_or_none(tracker: LLMUsageTracker) -> LLMUsageSummary | None:
    try:
        return tracker.summary()
    except RuntimeError:
        return None


def _validate_search_protocol(
    proposal: ImprovementProposal,
    *,
    evaluation_summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    directive = evaluation_summary.get("search_directive")
    if not isinstance(directive, Mapping):
        return
    required = directive.get("required_target_objectives")
    if isinstance(required, list) and (
        len(proposal.target_objectives) != len(required)
        or set(proposal.target_objectives) != set(required)
    ):
        raise ValueError(
            "Proposal target_objectives do not match the scheduled objectives: "
            f"{required}."
        )
    documentation_paths = [
        edit.relative_path
        for edit in proposal.edits
        if Path(edit.relative_path).suffix.lower() == ".md"
    ]
    if documentation_paths and not directive.get("documentation_edits_allowed", False):
        raise ValueError(
            "Documentation edits are not allowed for this objective slot: "
            f"{documentation_paths}."
        )
    threshold = directive.get("near_duplicate_similarity_threshold", 0.72)
    if not isinstance(threshold, (int, float)):
        raise ValueError("Invalid near-duplicate similarity threshold.")
    current_paths = {edit.relative_path for edit in proposal.edits}
    current_targets = set(proposal.target_objectives)
    for item in history:
        prior = item.get("proposal")
        if not isinstance(prior, Mapping):
            continue
        prior_paths = set(prior.get("edit_paths", []))
        prior_targets = set(prior.get("target_objectives", []))
        if not current_paths.intersection(prior_paths) or current_targets != prior_targets:
            continue
        similarity = _proposal_similarity(proposal, prior)
        if similarity >= float(threshold):
            raise ValueError(
                "Proposal is too similar to a prior hypothesis "
                f"(similarity={similarity:.3f}, threshold={float(threshold):.3f})."
            )


def _proposal_similarity(
    proposal: ImprovementProposal,
    prior: Mapping[str, Any],
) -> float:
    current = _meaningful_tokens(f"{proposal.title} {proposal.hypothesis}")
    previous = _meaningful_tokens(
        f"{prior.get('title', '')} {prior.get('hypothesis', '')}"
    )
    if not current or not previous:
        return 0.0
    return len(current.intersection(previous)) / len(current.union(previous))


def _meaningful_tokens(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "by",
        "for",
        "from",
        "if",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "will",
        "with",
        "without",
        "change",
        "changes",
        "improve",
        "pipeline",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", value.lower())
        if len(token) > 2 and token not in stopwords
    }
