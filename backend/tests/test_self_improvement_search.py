from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.agents.pipeline_improvement import (
    ImprovementEdit,
    ImprovementProposal,
    ImprovementProposalRecord,
    PipelineImprovementAgent,
    ProposalCall,
    ProposalGenerationError,
)
from backend.evaluation.objective_v3_schemas import V3EvaluationSummary
from backend.orchestration.adaptive_search import AdaptiveBranchingPolicy
from backend.orchestration.candidate_lifecycle import (
    CandidateEvaluationResult,
    apply_proposal,
    copy_root_candidate,
    editable_candidate_paths,
    rebind_evaluation_suites,
)
from backend.orchestration.greedy_search import ParetoGreedyPolicy
from backend.orchestration.pareto_archive_search import ParetoArchivePolicy
from backend.orchestration.pareto import ParetoArchive, pareto_relation
from backend.orchestration.self_improvement import (
    _proposal_directive,
    _proposal_history_for_config,
    run_self_improvement,
)
from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    ParetoTolerances,
    ProposalFocusAllocation,
    SelfImprovementConfig,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPOSITORY_ROOT / "project/datasets/reanalysis_era5_pressure_levels"
ROOT_CANDIDATE = (
    DATASET_DIR
    / "pipelines/pipeline-expert-direct-llm-stage-a-matched-r2-20260806t222100z"
)
BASE_SUITE = (
    DATASET_DIR
    / "benchmarks/constrained_v3_20260803/suites_public_policy_extensibility_v1"
    / "pipeline-expert-direct-llm-stage-a-matched-r2-20260806t222100z.json"
)


def _vector(
    materialization: float,
    throughput: float,
    output: int,
    quality: float,
) -> ObjectiveVector:
    return ObjectiveVector(
        materialization_seconds=materialization,
        consumer_samples_per_second=throughput,
        output_bytes=output,
        q_engineering=quality,
    )


def _summary(vector: ObjectiveVector) -> V3EvaluationSummary:
    return V3EvaluationSummary(
        target="candidate",
        constraints={
            "contract_correctness": "pass",
            "semantic_equivalence": "pass",
            "rerun_safety": "pass",
            "provenance_security": "pass",
        },
        feasible=True,
        engineering_assessed=True,
        objective_vector_complete=True,
        optimization_ready=True,
        thesis_evidence_ready=False,
        diagnostic_operational_metrics={
            "materialization_seconds": vector.materialization_seconds,
            "consumer_samples_per_second": vector.consumer_samples_per_second,
            "output_bytes": vector.output_bytes,
        },
        diagnostic_engineering_quality={"q_engineering": vector.q_engineering},
        operational_objectives={
            "materialization_seconds": vector.materialization_seconds,
            "consumer_samples_per_second": vector.consumer_samples_per_second,
            "output_bytes": vector.output_bytes,
        },
        engineering_objective={"q_engineering": vector.q_engineering},
        engineering_profile="general_pipeline",
    )


class _FakeProposer:
    model = "fake"

    def __init__(self) -> None:
        self.history_lengths: list[int] = []

    def propose(
        self,
        *,
        candidate_dir: Path,
        editable_paths: Sequence[str],
        evaluation_summary: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        max_edits: int = 4,
    ) -> ProposalCall:
        del evaluation_summary, max_edits
        self.history_lengths.append(len(history))
        self.assert_editable = "README.md" in editable_paths
        path = candidate_dir / "README.md"
        before = path.read_text(encoding="utf-8")
        after = before + "\nSearch experiment marker.\n"
        proposal = ImprovementProposal(
            title="Document one search iteration",
            hypothesis="A concise note should improve documentation evidence.",
            target_objectives=["q_engineering"],
            expected_tradeoffs="No operational behavior should change.",
            edits=[
                ImprovementEdit(
                    relative_path="README.md",
                    old_text=before,
                    new_text=after,
                    rationale="Exercise one exact candidate-owned edit.",
                )
            ],
            validation_plan="Run the frozen evaluator.",
        )
        record = ImprovementProposalRecord(
            model="fake",
            system_prompt_sha256="1" * 64,
            user_prompt_sha256="2" * 64,
            parent_source_sha256="3" * 64,
            proposal=proposal,
        )
        return ProposalCall(
            record=record,
            rendered_system_prompt="fake system",
            rendered_user_prompt="fake prompt",
        )


class _FakeEvaluator:
    def __init__(self) -> None:
        self.vectors = {
            "node_0000_root": _vector(1.0, 1_000.0, 10_000, 2.0),
            "node_0001": _vector(0.8, 1_000.0, 10_000, 2.0),
            "node_0002": _vector(0.9, 900.0, 11_000, 1.8),
        }

    def evaluate(
        self,
        *,
        node_id: str,
        candidate_dir: Path,
        node_dir: Path,
    ) -> CandidateEvaluationResult:
        self.last_candidate_dir = candidate_dir
        summary = _summary(self.vectors[node_id])
        report = node_dir / "fake_evaluation.json"
        report.write_text(
            json.dumps({"summary": summary.model_dump(mode="json")}, indent=2),
            encoding="utf-8",
        )
        return CandidateEvaluationResult(
            summary=summary,
            report_path=report,
            duration_seconds=0.01,
        )


class _InvalidProposalLLM:
    model = "gpt-5.5"

    def complete_json(self, **kwargs: Any) -> ImprovementProposal:
        kwargs["usage_tracker"].record_response(
            {
                "model": self.model,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 10},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 120,
                },
            },
            requested_model=self.model,
        )
        return kwargs["response_model"].model_validate({"title": "invalid"})


class _StaticProposalLLM:
    model = "gpt-5.5"

    def __init__(self, proposal: ImprovementProposal) -> None:
        self.proposal = proposal

    def complete_json(self, **kwargs: Any) -> ImprovementProposal:
        del kwargs
        return self.proposal


class SelfImprovementSearchTests(unittest.TestCase):
    def test_fifty_twenty_five_twenty_five_allocation_is_exact_and_interleaved(
        self,
    ) -> None:
        config = SelfImprovementConfig(
            run_id="test-focused-allocation",
            dataset_dir=DATASET_DIR,
            root_candidate_dir=ROOT_CANDIDATE,
            base_suite=BASE_SUITE,
            strategy="pareto_archive",
            max_iterations=20,
            pareto_objectives=["consumer_samples_per_second", "output_bytes"],
            proposal_focus_allocation=ProposalFocusAllocation(
                throughput=10,
                footprint=5,
                joint=5,
            ),
        )
        directives = [
            _proposal_directive(config, iteration=index, prior_proposals=[])
            for index in range(1, 21)
        ]
        focuses = [item["proposal_focus"] for item in directives]
        self.assertEqual(focuses.count("throughput"), 10)
        self.assertEqual(focuses.count("footprint"), 5)
        self.assertEqual(focuses.count("joint"), 5)
        self.assertEqual(
            focuses[:4], ["throughput", "footprint", "throughput", "joint"]
        )
        self.assertEqual(
            directives[3]["required_target_objectives"],
            ["consumer_samples_per_second", "output_bytes"],
        )

    def test_operational_only_vector_does_not_require_engineering_judge(self) -> None:
        summary = V3EvaluationSummary(
            target="candidate",
            constraints={
                "contract_correctness": "pass",
                "semantic_equivalence": "pass",
                "rerun_safety": "pass",
                "provenance_security": "pass",
            },
            feasible=True,
            engineering_assessed=False,
            objective_vector_complete=False,
            optimization_ready=False,
            thesis_evidence_ready=False,
            diagnostic_operational_metrics={
                "materialization_seconds": 1.0,
                "consumer_samples_per_second": 42.0,
                "output_bytes": 1000,
            },
            operational_objectives={
                "materialization_seconds": 1.0,
                "consumer_samples_per_second": 42.0,
                "output_bytes": 1000,
            },
            engineering_profile="general_pipeline",
        )
        vector = ObjectiveVector.from_summary(
            summary,
            required_objectives=["consumer_samples_per_second", "output_bytes"],
        )
        self.assertIsNone(vector.q_engineering)
        self.assertEqual(vector.consumer_samples_per_second, 42.0)

    def test_pareto_archive_uses_only_declared_objectives(self) -> None:
        objectives = ["consumer_samples_per_second", "output_bytes"]
        policy = ParetoArchivePolicy(
            ParetoTolerances(), objectives=objectives, seed=7
        )
        root = _vector(1.0, 100.0, 1000, 4.0)
        policy.initialize("root", root)
        selection = policy.select_parent()
        child = _vector(2.0, 120.0, 900, 0.0)
        decision = policy.observe(selection, "child", child)
        self.assertTrue(decision.admitted_to_archive)
        self.assertEqual(policy.state().archive_ids, ["child"])

    def test_proposal_history_mode_is_explicit_and_defaults_to_full(self) -> None:
        config = SelfImprovementConfig(
            run_id="test-history-mode",
            dataset_dir=DATASET_DIR,
            root_candidate_dir=ROOT_CANDIDATE,
            base_suite=BASE_SUITE,
            strategy="pareto_greedy",
        )
        self.assertEqual(config.proposal_history_mode, "full")
        hidden = config.model_copy(update={"proposal_history_mode": "none"})
        self.assertEqual(_proposal_history_for_config(hidden, {}, {}), [])

    def test_objective_cycle_and_documentation_budget_are_deterministic(self) -> None:
        config = SelfImprovementConfig(
            run_id="test-objective-cycle",
            dataset_dir=DATASET_DIR,
            root_candidate_dir=ROOT_CANDIDATE,
            base_suite=BASE_SUITE,
            strategy="pareto_greedy",
        )
        expected = [
            ["materialization_seconds", "consumer_samples_per_second"],
            ["output_bytes"],
            ["materialization_seconds"],
            ["q_engineering"],
            ["consumer_samples_per_second"],
        ]
        observed = [
            _proposal_directive(config, iteration=index, prior_proposals=[])[
                "required_target_objectives"
            ]
            for index in range(1, 6)
        ]
        self.assertEqual(observed, expected)
        first_quality = _proposal_directive(
            config, iteration=4, prior_proposals=[]
        )
        second_quality = _proposal_directive(
            config,
            iteration=4,
            prior_proposals=[{"has_documentation_edits": True}],
        )
        self.assertTrue(first_quality["documentation_edits_allowed"])
        self.assertFalse(second_quality["documentation_edits_allowed"])

    def test_agent_enforces_objective_documentation_and_novelty_directives(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            candidate = Path(temporary)
            (candidate / "README.md").write_text("Current guide.\n", encoding="utf-8")
            proposal = ImprovementProposal(
                title="Rewrite the current guide",
                hypothesis="Changing the guide should make the workflow easier to use.",
                target_objectives=["materialization_seconds"],
                expected_tradeoffs="No executable behavior changes.",
                edits=[
                    ImprovementEdit(
                        relative_path="README.md",
                        old_text="Current guide.",
                        new_text="Expanded guide.",
                        rationale="Add documentation.",
                    )
                ],
                validation_plan="Inspect the guide.",
            )
            agent = PipelineImprovementAgent(_StaticProposalLLM(proposal))
            evidence = {
                "search_directive": {
                    "required_target_objectives": ["materialization_seconds"],
                    "documentation_edits_allowed": False,
                    "near_duplicate_similarity_threshold": 0.72,
                }
            }
            with self.assertRaisesRegex(
                ProposalGenerationError, "Documentation edits are not allowed"
            ):
                agent.propose(
                    candidate_dir=candidate,
                    editable_paths=["README.md"],
                    evaluation_summary=evidence,
                    history=[],
                )

            evidence["search_directive"]["documentation_edits_allowed"] = True
            prior = {
                "proposal": {
                    "title": proposal.title,
                    "hypothesis": proposal.hypothesis,
                    "target_objectives": proposal.target_objectives,
                    "edit_paths": ["README.md"],
                    "has_documentation_edits": True,
                }
            }
            with self.assertRaisesRegex(
                ProposalGenerationError, "too similar to a prior hypothesis"
            ):
                agent.propose(
                    candidate_dir=candidate,
                    editable_paths=["README.md"],
                    evaluation_summary=evidence,
                    history=[prior],
                )

    def test_pareto_comparison_respects_all_four_directions_and_noise(self) -> None:
        tolerances = ParetoTolerances()
        baseline = _vector(1.0, 1_000.0, 10_000, 2.0)
        better = _vector(0.8, 1_100.0, 9_000, 2.3)
        noisy = _vector(0.99, 1_020.0, 9_970, 2.05)
        tradeoff = _vector(0.8, 900.0, 9_000, 2.3)

        self.assertEqual(pareto_relation(better, baseline, tolerances), "dominates")
        self.assertEqual(pareto_relation(noisy, baseline, tolerances), "equivalent")
        self.assertEqual(pareto_relation(tradeoff, baseline, tolerances), "tradeoff")

        archive = ParetoArchive(tolerances)
        archive.add_root("root", baseline)
        decision = archive.consider("better", better)
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.removed_ids, ("root",))

    def test_greedy_advances_only_to_new_frontier_candidates(self) -> None:
        policy = ParetoGreedyPolicy(ParetoTolerances())
        root = _vector(1.0, 1_000.0, 10_000, 2.0)
        policy.initialize("root", root)
        first = policy.select_parent()
        admitted = policy.observe(
            first, "child-1", _vector(0.8, 1_000.0, 10_000, 2.0)
        )
        self.assertTrue(admitted.admitted_to_archive)
        self.assertEqual(policy.select_parent().node_id, "child-1")
        rejected = policy.observe(
            policy.select_parent(),
            "child-2",
            _vector(0.9, 900.0, 11_000, 1.8),
        )
        self.assertFalse(rejected.admitted_to_archive)
        self.assertEqual(policy.select_parent().node_id, "child-1")

    def test_adaptive_switches_after_two_valid_nonimprovements(self) -> None:
        policy = AdaptiveBranchingPolicy(
            ParetoTolerances(), stagnation_patience=2, branch_width=2
        )
        policy.initialize("root", _vector(1.0, 1_000.0, 10_000, 2.0))
        for index in (1, 2):
            selection = policy.select_parent()
            policy.observe(
                selection,
                f"rejected-{index}",
                _vector(1.2 + index / 10, 800.0, 12_000, 1.5),
            )
        state = policy.state()
        self.assertEqual(state.phase, "branching")
        self.assertEqual(len(state.branches), 2)
        parents = [policy.select_parent(), policy.select_parent()]
        self.assertEqual({item.branch_id for item in parents}, {"branch_01", "branch_02"})
        self.assertEqual(len({item.node_id for item in parents}), 2)

    def test_candidate_patch_and_suite_rebinding_preserve_boundaries(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            root = Path(temporary)
            parent = root / "parent"
            paths = copy_root_candidate(
                base_suite=BASE_SUITE,
                repository_root=REPOSITORY_ROOT,
                destination=parent,
            )
            child = root / "child"
            from backend.orchestration.candidate_lifecycle import copy_child_candidate

            copy_child_candidate(parent, child)
            editable = editable_candidate_paths(paths)
            before = (child / "README.md").read_text(encoding="utf-8")
            proposal = ImprovementProposal(
                title="Clarify one command",
                hypothesis="One corrected command improves concise documentation.",
                target_objectives=["q_engineering"],
                expected_tradeoffs="None.",
                edits=[
                    ImprovementEdit(
                        relative_path="README.md",
                        old_text=before,
                        new_text=before + "\nVerified command.\n",
                        rationale="Improve one stale instruction.",
                    )
                ],
                validation_plan="Rebind and validate both frozen suites.",
            )
            protected_before = hashlib.sha256(
                (child / "pipeline_contract.json").read_bytes()
            ).hexdigest()
            apply_proposal(
                candidate_dir=child,
                parent_dir=parent,
                proposal=proposal,
                editable_paths=editable,
                patch_path=root / "patch.diff",
            )
            self.assertEqual(
                protected_before,
                hashlib.sha256((child / "pipeline_contract.json").read_bytes()).hexdigest(),
            )
            suites = rebind_evaluation_suites(
                base_suite=BASE_SUITE,
                candidate_dir=child,
                candidate_id="node-test",
                output_dir=root / "inputs",
                repository_root=REPOSITORY_ROOT,
                require_extensibility=True,
            )
            primary = json.loads(suites.primary.read_text(encoding="utf-8"))
            assert suites.alternate is not None
            alternate = json.loads(suites.alternate.read_text(encoding="utf-8"))
            self.assertEqual(
                primary["candidate_id"],
                "pipeline-expert-direct-llm-stage-a-matched-r2-20260806t222100z",
            )
            self.assertEqual(primary["candidate_id"], alternate["candidate_id"])
            self.assertTrue(primary["suite_id"].endswith("__node-test"))
            self.assertEqual(
                primary["candidate_source"], alternate["candidate_source"]
            )
            self.assertIn("runs/self_improvement", str(root / "runs/self_improvement"))

    def test_protected_file_check_allows_an_absent_optional_requirements_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            for relative, content in {
                "manifest.json": "{}\n",
                "pipeline_contract.json": "{}\n",
                "run_pipeline.py": "print('run')\n",
                "pyproject.toml": "[project]\nname = 'candidate'\nversion = '0.1.0'\n",
                "README.md": "Candidate\n",
            }.items():
                (parent / relative).write_text(content, encoding="utf-8")
            child = root / "child"
            from backend.orchestration.candidate_lifecycle import copy_child_candidate

            copy_child_candidate(parent, child)
            proposal = ImprovementProposal(
                title="Clarify candidate",
                hypothesis="A source-only edit should preserve protected files.",
                target_objectives=["q_engineering"],
                expected_tradeoffs="None.",
                edits=[
                    ImprovementEdit(
                        relative_path="README.md",
                        old_text="Candidate\n",
                        new_text="Candidate\n\nClarified.\n",
                        rationale="Exercise protected-file validation.",
                    )
                ],
                validation_plan="Validate source invariants.",
            )
            apply_proposal(
                candidate_dir=child,
                parent_dir=parent,
                proposal=proposal,
                editable_paths=["README.md", "pyproject.toml"],
                patch_path=root / "patch.diff",
            )
            self.assertFalse((parent / "requirements.txt").exists())
            self.assertFalse((child / "requirements.txt").exists())

    def test_multi_file_patch_is_atomic_when_a_later_edit_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            root = Path(temporary)
            parent = root / "parent"
            paths = copy_root_candidate(
                base_suite=BASE_SUITE,
                repository_root=REPOSITORY_ROOT,
                destination=parent,
            )
            child = root / "child"
            from backend.orchestration.candidate_lifecycle import copy_child_candidate

            copy_child_candidate(parent, child)
            editable = editable_candidate_paths(paths)
            readme_before = (child / "README.md").read_text(encoding="utf-8")
            invalid_path = next(path for path in editable if path != "README.md")
            proposal = ImprovementProposal(
                title="Exercise atomic patch validation",
                hypothesis="No files should change when any exact replacement is invalid.",
                target_objectives=["q_engineering"],
                expected_tradeoffs="None because evaluation must not start.",
                edits=[
                    ImprovementEdit(
                        relative_path="README.md",
                        old_text=readme_before,
                        new_text=readme_before + "\nThis must not be written.\n",
                        rationale="This valid edit is intentionally staged first.",
                    ),
                    ImprovementEdit(
                        relative_path=invalid_path,
                        old_text="an exact string that is absent from the candidate",
                        new_text="replacement",
                        rationale="Force validation to fail after the first edit is staged.",
                    ),
                ],
                validation_plan="Assert the candidate remains byte-for-byte unchanged.",
            )

            with self.assertRaisesRegex(ValueError, "found 0"):
                apply_proposal(
                    candidate_dir=child,
                    parent_dir=parent,
                    proposal=proposal,
                    editable_paths=editable,
                    patch_path=root / "patch.diff",
                )

            self.assertEqual(
                (child / "README.md").read_text(encoding="utf-8"), readme_before
            )
            self.assertFalse((root / "patch.diff").exists())

    def test_same_file_edits_are_ordered_and_terminal_newlines_are_normalized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            root = Path(temporary)
            parent = root / "parent"
            paths = copy_root_candidate(
                base_suite=BASE_SUITE,
                repository_root=REPOSITORY_ROOT,
                destination=parent,
            )
            child = root / "child"
            from backend.orchestration.candidate_lifecycle import copy_child_candidate

            copy_child_candidate(parent, child)
            editable = editable_candidate_paths(paths)
            before = (child / "README.md").read_text(encoding="utf-8")
            first = before.replace(
                "# ERA5 pressure-level family adapter",
                "# Verified ERA5 pressure-level family adapter",
                1,
            )
            proposal = ImprovementProposal(
                title="Apply two ordered documentation corrections",
                hypothesis="Ordered replacements should remain one atomic source patch.",
                target_objectives=["q_engineering"],
                expected_tradeoffs="No executable behavior changes.",
                edits=[
                    ImprovementEdit(
                        relative_path="README.md",
                        old_text=before + "\n",
                        new_text=first,
                        rationale="Exercise terminal-newline normalization.",
                    ),
                    ImprovementEdit(
                        relative_path="README.md",
                        old_text="Credentials are needed only for cache misses.",
                        new_text="Provider credentials are needed only for cache misses.",
                        rationale="Exercise an ordered second edit in the same file.",
                    ),
                ],
                validation_plan="Inspect one combined diff and the final README.",
            )

            apply_proposal(
                candidate_dir=child,
                parent_dir=parent,
                proposal=proposal,
                editable_paths=editable,
                patch_path=root / "patch.diff",
            )

            after = (child / "README.md").read_text(encoding="utf-8")
            self.assertIn("# Verified ERA5", after)
            self.assertIn("Provider credentials are needed", after)
            self.assertEqual((root / "patch.diff").read_text().count("--- a/"), 1)

    def test_failed_proposal_persists_billed_usage_and_prompt_provenance(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            dataset = Path(temporary) / "dataset"
            dataset.mkdir()
            config = SelfImprovementConfig(
                run_id="test-failed-proposal-usage",
                dataset_dir=dataset,
                root_candidate_dir=ROOT_CANDIDATE,
                base_suite=BASE_SUITE,
                strategy="pareto_greedy",
                max_iterations=1,
                max_wall_time_seconds=60,
            )
            report, _ = run_self_improvement(
                config,
                proposer=PipelineImprovementAgent(_InvalidProposalLLM()),
                evaluator=_FakeEvaluator(),
                repository_root=REPOSITORY_ROOT,
            )

            node = json.loads(Path(report.node_records[1]).read_text())
            failure_path = Path(node["proposal_record"])
            failure = json.loads(failure_path.read_text())
            self.assertEqual(node["status"], "proposal_failed")
            self.assertEqual(failure["llm_usage"]["call_count"], 1)
            self.assertAlmostEqual(
                failure["llm_usage"]["estimated_cost_usd"], 0.001055
            )
            self.assertTrue((failure_path.parent / "proposal_system.md").is_file())
            self.assertTrue((failure_path.parent / "proposal_context.md").is_file())

    def test_full_greedy_run_keeps_nodes_out_of_general_pipelines(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            dataset = Path(temporary) / "dataset"
            dataset.mkdir()
            config = SelfImprovementConfig(
                run_id="test-greedy-run",
                dataset_dir=dataset,
                root_candidate_dir=ROOT_CANDIDATE,
                base_suite=BASE_SUITE,
                strategy="pareto_greedy",
                max_iterations=2,
                max_wall_time_seconds=60,
            )
            proposer = _FakeProposer()
            report, path = run_self_improvement(
                config,
                proposer=proposer,
                evaluator=_FakeEvaluator(),
                repository_root=REPOSITORY_ROOT,
            )

            self.assertEqual(report.status, "completed")
            self.assertEqual(report.continuation_node_id, "node_0001")
            self.assertEqual(report.pareto_archive, ["node_0001"])
            self.assertEqual(len(report.node_records), 3)
            self.assertFalse((dataset / "pipelines").exists())
            self.assertTrue(
                (
                    dataset
                    / "runs/self_improvement/test-greedy-run/nodes/node_0002/candidate"
                ).is_dir()
            )
            self.assertTrue(path.is_file())
            self.assertTrue(proposer.assert_editable)
            self.assertEqual(proposer.history_lengths, [1, 2])

    def test_no_history_ablation_hides_all_prior_nodes_from_proposer(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as temporary:
            dataset = Path(temporary) / "dataset"
            dataset.mkdir()
            config = SelfImprovementConfig(
                run_id="test-no-history-run",
                dataset_dir=dataset,
                root_candidate_dir=ROOT_CANDIDATE,
                base_suite=BASE_SUITE,
                strategy="pareto_greedy",
                max_iterations=2,
                max_wall_time_seconds=60,
                proposal_history_mode="none",
            )
            proposer = _FakeProposer()
            report, _ = run_self_improvement(
                config,
                proposer=proposer,
                evaluator=_FakeEvaluator(),
                repository_root=REPOSITORY_ROOT,
            )

            self.assertEqual(report.status, "completed")
            self.assertEqual(proposer.history_lengths, [0, 0])


if __name__ == "__main__":
    unittest.main()
