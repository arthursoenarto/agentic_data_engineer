"""Pipeline-level self-improvement over immutable, externally evaluated candidates."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from backend.agents.pipeline_improvement import (
    ImprovementProposal,
    PipelineImprovementAgent,
    ProposalCall,
    ProposalGenerationError,
)
from backend.llm import LLMClient
from backend.orchestration.adaptive_search import AdaptiveBranchingPolicy
from backend.orchestration.candidate_lifecycle import (
    CandidateEvaluationResult,
    FrozenSuiteCandidateEvaluator,
    apply_proposal,
    base_suite_candidate_dir,
    base_suite_search_invariants,
    candidate_source_paths,
    copy_child_candidate,
    copy_root_candidate,
    editable_candidate_paths,
)
from backend.orchestration.greedy_search import ParetoGreedyPolicy
from backend.orchestration.pareto import objective_changes, pareto_relation
from backend.orchestration.pareto_archive_search import ParetoArchivePolicy
from backend.orchestration.search_policy import SearchPolicy
from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    SearchNodeRecord,
    SelfImprovementConfig,
    SelfImprovementRunReport,
    compact_history_entry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CandidateProposer(Protocol):
    def propose(
        self,
        *,
        candidate_dir: Path,
        editable_paths: Sequence[str],
        evaluation_summary: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        max_edits: int = 4,
    ) -> ProposalCall: ...


class CandidateEvaluator(Protocol):
    def evaluate(
        self,
        *,
        node_id: str,
        candidate_dir: Path,
        node_dir: Path,
    ) -> CandidateEvaluationResult: ...


def run_self_improvement(
    config: SelfImprovementConfig,
    *,
    proposer: CandidateProposer | None = None,
    evaluator: CandidateEvaluator | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[SelfImprovementRunReport, Path]:
    """Run one bounded search and preserve every source/evaluation node."""

    repository_root = repository_root.resolve()
    dataset_dir = _repo_path(config.dataset_dir, repository_root)
    root_candidate_dir = _repo_path(config.root_candidate_dir, repository_root)
    base_suite = _repo_path(config.base_suite, repository_root)
    pinned_root = base_suite_candidate_dir(base_suite, repository_root)
    if pinned_root != root_candidate_dir:
        raise ValueError(
            "root_candidate_dir must equal the source cwd pinned by base_suite."
        )
    search_invariants = base_suite_search_invariants(base_suite)
    run_dir = dataset_dir / "runs" / "self_improvement" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    nodes_dir = run_dir / "nodes"
    nodes_dir.mkdir()
    config_path = run_dir / "config.json"
    _write_new(config_path, config.model_dump_json(indent=2) + "\n")
    report_path = run_dir / "search_run.json"
    journal_path = run_dir / "journal.jsonl"
    report = SelfImprovementRunReport(
        run_id=config.run_id,
        strategy=config.strategy,
        started_at=_now(),
        config_path=_display(config_path, repository_root),
    )
    _write_report(report_path, report)
    started = time.monotonic()
    records: dict[str, SearchNodeRecord] = {}
    summaries: dict[str, Mapping[str, Any]] = {}
    proposal_summaries: dict[str, Mapping[str, Any]] = {}

    try:
        llm: LLMClient | None = None
        if proposer is None or evaluator is None:
            llm = LLMClient(
                model=config.llm_model,
                timeout_seconds=config.llm_timeout_seconds,
            )
        resolved_proposer = proposer or PipelineImprovementAgent(llm)  # type: ignore[arg-type]
        resolved_evaluator = evaluator or FrozenSuiteCandidateEvaluator(
            repository_root=repository_root,
            dataset_dir=dataset_dir,
            base_suite=base_suite,
            llm_client=llm,  # type: ignore[arg-type]
            llm_timeout_seconds=config.llm_timeout_seconds,
            require_extensibility=config.require_extensibility_probe,
            operational_repetitions=config.operational_repetitions,
            search_run_id=config.run_id,
            required_objectives=config.pareto_objectives,
        )
        policy = _policy(config)
    except Exception as error:  # noqa: BLE001 - persist initialization failure.
        report.status = "failed"
        report.stop_reason = "workflow_error"
        report.error = f"{type(error).__name__}: {error}"
        report.completed_at = _now()
        _write_report(report_path, report)
        return report, report_path

    try:
        root_id = "node_0000_root"
        root_node_dir = nodes_dir / root_id
        root_node_dir.mkdir()
        root_candidate = root_node_dir / "candidate"
        source_paths = copy_root_candidate(
            base_suite=base_suite,
            repository_root=repository_root,
            destination=root_candidate,
        )
        editable_paths = editable_candidate_paths(source_paths)
        root_started = _now()
        root_result = resolved_evaluator.evaluate(
            node_id=root_id,
            candidate_dir=root_candidate,
            node_dir=root_node_dir,
        )
        root_record = _evaluated_record(
            node_id=root_id,
            parent=None,
            iteration=0,
            depth=0,
            phase="root",
            branch_id=None,
            candidate_dir=root_candidate,
            proposal_record=None,
            result=root_result,
            policy=None,
            started_at=root_started,
            repository_root=repository_root,
            tolerances=config.tolerances,
            pareto_objectives=config.pareto_objectives,
        )
        if root_record.objective is not None:
            policy.initialize(root_id, root_record.objective)
            root_record.admitted_to_archive = True
            root_record.policy_reason = "Initialized the Pareto archive."
        _persist_node(
            root_node_dir,
            root_record,
            report=report,
            report_path=report_path,
            journal_path=journal_path,
            repository_root=repository_root,
        )
        records[root_id] = root_record
        if root_result.summary is not None:
            summaries[root_id] = _proposal_evidence(root_result)
        report.root_node_id = root_id
        if root_record.objective is not None:
            _sync_policy(report, policy)
        _write_report(report_path, report)
        if root_record.objective is None:
            report.status = "failed"
            report.stop_reason = "root_not_optimization_ready"
            report.completed_at = _now()
            _write_report(report_path, report)
            return report, report_path

        for iteration in range(1, config.max_iterations + 1):
            if time.monotonic() - started >= config.max_wall_time_seconds:
                report.stop_reason = "wall_time_budget"
                break
            selection = policy.select_parent()
            parent = records[selection.node_id]
            node_id = f"node_{iteration:04d}"
            node_dir = nodes_dir / node_id
            node_dir.mkdir()
            candidate_dir = node_dir / "candidate"
            copy_child_candidate(
                Path(_absolute(parent.candidate_dir, repository_root)),
                candidate_dir,
            )
            node_started = _now()
            proposal_path: Path | None = None
            try:
                directive = _proposal_directive(
                    config,
                    iteration=iteration,
                    prior_proposals=proposal_summaries.values(),
                )
                call = resolved_proposer.propose(
                    candidate_dir=Path(
                        _absolute(parent.candidate_dir, repository_root)
                    ),
                    editable_paths=editable_paths,
                    evaluation_summary={
                        "evaluation_evidence": summaries[selection.node_id],
                        "frozen_search_invariants": search_invariants,
                        "search_directive": directive,
                    },
                    history=_proposal_history_for_config(
                        config,
                        records,
                        proposal_summaries,
                    ),
                    max_edits=config.max_edits_per_proposal,
                )
                proposal_summaries[node_id] = _proposal_summary(
                    call.record.proposal
                )
                proposal_path = node_dir / "proposal.json"
                _write_new(
                    proposal_path,
                    call.record.model_dump_json(indent=2) + "\n",
                )
                _write_new(
                    node_dir / "proposal_system.md",
                    call.rendered_system_prompt + "\n",
                )
                _write_new(node_dir / "proposal_context.md", call.rendered_user_prompt + "\n")
            except Exception as error:  # noqa: BLE001 - durable failed proposal node.
                if isinstance(error, ProposalGenerationError):
                    proposal_path = node_dir / "proposal_failure.json"
                    _write_new(
                        proposal_path,
                        error.record.model_dump_json(indent=2) + "\n",
                    )
                    _write_new(
                        node_dir / "proposal_system.md",
                        error.rendered_system_prompt + "\n",
                    )
                    _write_new(
                        node_dir / "proposal_context.md",
                        error.rendered_user_prompt + "\n",
                    )
                record = _failure_record(
                    node_id=node_id,
                    parent=parent,
                    iteration=iteration,
                    selection=selection,
                    status="proposal_failed",
                    candidate_dir=candidate_dir,
                    failure=f"{type(error).__name__}: {error}",
                    started_at=node_started,
                    repository_root=repository_root,
                    proposal_record=proposal_path,
                )
                observation = policy.observe(selection, node_id, None)
                record.policy_reason = observation.reason
                _persist_node(
                    node_dir,
                    record,
                    report=report,
                    report_path=report_path,
                    journal_path=journal_path,
                    repository_root=repository_root,
                )
                records[node_id] = record
                _sync_policy(report, policy)
                _write_report(report_path, report)
                continue

            try:
                apply_proposal(
                    candidate_dir=candidate_dir,
                    parent_dir=Path(_absolute(parent.candidate_dir, repository_root)),
                    proposal=call.record.proposal,
                    editable_paths=editable_paths,
                    patch_path=node_dir / "patch.diff",
                )
            except Exception as error:  # noqa: BLE001 - durable invalid-patch node.
                record = _failure_record(
                    node_id=node_id,
                    parent=parent,
                    iteration=iteration,
                    selection=selection,
                    status="invalid_patch",
                    candidate_dir=candidate_dir,
                    failure=f"{type(error).__name__}: {error}",
                    started_at=node_started,
                    repository_root=repository_root,
                    proposal_record=proposal_path,
                )
                observation = policy.observe(selection, node_id, None)
                record.policy_reason = observation.reason
                _persist_node(
                    node_dir,
                    record,
                    report=report,
                    report_path=report_path,
                    journal_path=journal_path,
                    repository_root=repository_root,
                )
                records[node_id] = record
                _sync_policy(report, policy)
                _write_report(report_path, report)
                continue

            result = resolved_evaluator.evaluate(
                node_id=node_id,
                candidate_dir=candidate_dir,
                node_dir=node_dir,
            )
            record = _evaluated_record(
                node_id=node_id,
                parent=parent,
                iteration=iteration,
                depth=parent.depth + 1,
                phase=selection.phase,  # type: ignore[arg-type]
                branch_id=selection.branch_id,
                candidate_dir=candidate_dir,
                proposal_record=proposal_path,
                result=result,
                policy=policy,
                started_at=node_started,
                repository_root=repository_root,
                tolerances=config.tolerances,
                pareto_objectives=config.pareto_objectives,
            )
            observation = policy.observe(selection, node_id, record.objective)
            record.admitted_to_archive = observation.admitted_to_archive
            record.policy_reason = observation.reason
            _persist_node(
                node_dir,
                record,
                report=report,
                report_path=report_path,
                journal_path=journal_path,
                repository_root=repository_root,
            )
            records[node_id] = record
            if result.summary is not None:
                summaries[node_id] = _proposal_evidence(result)
            _sync_policy(report, policy)
            _write_report(report_path, report)
        else:
            report.stop_reason = "iteration_budget"

        if report.stop_reason is None:
            report.stop_reason = "iteration_budget"
        report.status = "completed"
        report.completed_at = _now()
        _sync_policy(report, policy)
        _write_report(report_path, report)
        return report, report_path
    except Exception as error:  # noqa: BLE001 - preserve terminal workflow evidence.
        report.status = "failed"
        report.stop_reason = "workflow_error"
        report.error = f"{type(error).__name__}: {error}"
        report.completed_at = _now()
        _write_report(report_path, report)
        return report, report_path


def _policy(config: SelfImprovementConfig) -> SearchPolicy:
    if config.strategy == "pareto_greedy":
        return ParetoGreedyPolicy(config.tolerances, config.pareto_objectives)
    if config.strategy == "pareto_archive":
        return ParetoArchivePolicy(
            config.tolerances,
            objectives=config.pareto_objectives,
            seed=config.parent_selection_seed,
        )
    return AdaptiveBranchingPolicy(
        config.tolerances,
        stagnation_patience=config.stagnation_patience,
        branch_width=config.branch_width,
        objectives=config.pareto_objectives,
    )


def _evaluated_record(
    *,
    node_id: str,
    parent: SearchNodeRecord | None,
    iteration: int,
    depth: int,
    phase: str,
    branch_id: str | None,
    candidate_dir: Path,
    proposal_record: Path | None,
    result: CandidateEvaluationResult,
    policy: SearchPolicy | None,
    started_at: datetime,
    repository_root: Path,
    tolerances: Any,
    pareto_objectives: Sequence[str],
) -> SearchNodeRecord:
    summary = result.summary
    objective = (
        ObjectiveVector.from_summary(
            summary,
            required_objectives=list(pareto_objectives),
        )
        if summary is not None
        and summary.feasible
        and set(pareto_objectives).issubset(
            set(summary.operational_objectives) | set(summary.engineering_objective)
        )
        else None
    )
    status = (
        "evaluation_failed"
        if summary is None
        else (
            "optimization_ready"
            if objective is not None
            else ("incomplete" if summary.feasible else "infeasible")
        )
    )
    relation = "not_comparable"
    changes: dict[str, float] = {}
    if parent is not None and parent.objective is not None and objective is not None:
        relation = pareto_relation(
            objective,
            parent.objective,
            tolerances,
            pareto_objectives,
        )
        changes = objective_changes(objective, parent.objective, pareto_objectives)
    constraints = (
        {
            name: str(value.value if hasattr(value, "value") else value)
            for name, value in summary.constraints.items()
        }
        if summary is not None
        else {}
    )
    return SearchNodeRecord(
        node_id=node_id,
        parent_id=parent.node_id if parent else None,
        iteration=iteration,
        depth=depth,
        phase=phase,  # type: ignore[arg-type]
        branch_id=branch_id,
        status=status,
        candidate_dir=_display(candidate_dir, repository_root),
        proposal_record=(
            _display(proposal_record, repository_root) if proposal_record else None
        ),
        evaluation_report=_display(result.report_path, repository_root),
        evaluation_seconds=result.duration_seconds,
        constraints=constraints,
        optimization_ready=objective is not None,
        objective=objective,
        parent_relation=relation,
        objective_changes=changes,
        failure=result.failure,
        created_at=started_at,
        completed_at=_now(),
    )


def _failure_record(
    *,
    node_id: str,
    parent: SearchNodeRecord,
    iteration: int,
    selection: Any,
    status: str,
    candidate_dir: Path,
    failure: str,
    started_at: datetime,
    repository_root: Path,
    proposal_record: Path | None = None,
) -> SearchNodeRecord:
    return SearchNodeRecord(
        node_id=node_id,
        parent_id=parent.node_id,
        iteration=iteration,
        depth=parent.depth + 1,
        phase=selection.phase,
        branch_id=selection.branch_id,
        status=status,
        candidate_dir=_display(candidate_dir, repository_root),
        proposal_record=(
            _display(proposal_record, repository_root) if proposal_record else None
        ),
        failure=failure,
        created_at=started_at,
        completed_at=_now(),
    )


def _persist_node(
    node_dir: Path,
    node: SearchNodeRecord,
    *,
    report: SelfImprovementRunReport,
    report_path: Path,
    journal_path: Path,
    repository_root: Path,
) -> None:
    node_path = node_dir / "node.json"
    _write_new(node_path, node.model_dump_json(indent=2) + "\n")
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(node.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    report.node_records.append(_display(node_path, repository_root))
    _write_report(report_path, report)


def _sync_policy(report: SelfImprovementRunReport, policy: SearchPolicy) -> None:
    state = policy.state()
    report.policy_state = state
    report.pareto_archive = state.archive_ids
    report.continuation_node_id = state.incumbent_id


def _write_report(path: Path, report: SelfImprovementRunReport) -> None:
    content = report.model_dump_json(indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def _repo_path(path: Path, repository_root: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"Search path escapes repository: {resolved}") from error
    return resolved


def _display(path: Path, repository_root: Path) -> str:
    return str(path.resolve().relative_to(repository_root.resolve()))


def _absolute(raw: str, repository_root: Path) -> str:
    path = Path(raw)
    return str(path if path.is_absolute() else repository_root / path)


def _now() -> datetime:
    return datetime.now(UTC)


def _proposal_evidence(result: CandidateEvaluationResult) -> dict[str, Any]:
    if result.summary is None:
        raise ValueError("Proposal evidence requires an evaluation summary.")
    return {
        "summary": result.summary.model_dump(mode="json"),
        "detailed_feedback": dict(result.proposal_feedback or {}),
        "evaluation_report": str(result.report_path),
    }


def _proposal_directive(
    config: SelfImprovementConfig,
    *,
    iteration: int,
    prior_proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    focus = None
    if config.proposal_focus_allocation is not None:
        focus_order: list[tuple[str, list[str]]] = []
        remaining = {
            "throughput": config.proposal_focus_allocation.throughput,
            "footprint": config.proposal_focus_allocation.footprint,
            "joint": config.proposal_focus_allocation.joint,
        }
        mapping = {
            "throughput": ["consumer_samples_per_second"],
            "footprint": ["output_bytes"],
            "joint": ["consumer_samples_per_second", "output_bytes"],
        }
        # Deterministic weighted round-robin keeps each focus distributed over time.
        while any(remaining.values()):
            for name in ("throughput", "footprint", "throughput", "joint"):
                if remaining[name] > 0:
                    focus_order.append((name, mapping[name]))
                    remaining[name] -= 1
        focus, targets = focus_order[iteration - 1]
    else:
        cycle = config.proposal_objective_cycle
        targets = cycle[(iteration - 1) % len(cycle)]
    documentation_used = sum(
        bool(item.get("has_documentation_edits")) for item in prior_proposals
    )
    documentation_allowed = (
        "q_engineering" in targets
        and documentation_used < config.max_documentation_proposals
    )
    return {
        "iteration": iteration,
        "proposal_focus": focus,
        "required_target_objectives": targets,
        "documentation_edits_allowed": documentation_allowed,
        "documentation_proposals_used": documentation_used,
        "documentation_proposal_limit": config.max_documentation_proposals,
        "near_duplicate_similarity_threshold": config.proposal_novelty_threshold,
        "rules": [
            "target_objectives must exactly match required_target_objectives",
            "do not edit Markdown unless documentation_edits_allowed is true",
            "propose a causal mechanism distinct from all prior proposals",
            "use operational_evidence when targeting operational objectives",
        ],
    }


def _proposal_summary(proposal: ImprovementProposal) -> dict[str, Any]:
    paths = [edit.relative_path for edit in proposal.edits]
    return {
        "title": proposal.title,
        "hypothesis": proposal.hypothesis,
        "target_objectives": proposal.target_objectives,
        "edit_paths": paths,
        "has_documentation_edits": any(Path(path).suffix == ".md" for path in paths),
    }


def _proposal_history(
    records: Mapping[str, SearchNodeRecord],
    proposals: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    history: list[Mapping[str, Any]] = []
    for node_id, record in records.items():
        entry = dict(compact_history_entry(record))
        if node_id in proposals:
            entry["proposal"] = proposals[node_id]
        history.append(entry)
    return history


def _proposal_history_for_config(
    config: SelfImprovementConfig,
    records: Mapping[str, SearchNodeRecord],
    proposals: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Expose trajectory evidence according to the frozen ablation protocol."""

    if config.proposal_history_mode == "none":
        return []
    return _proposal_history(records, proposals)
