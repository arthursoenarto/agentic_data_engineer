from __future__ import annotations

import unittest

from backend.evaluation.check_promotion import (
    GeneratedCheckCandidate,
    assess_generated_check_candidate,
)


def _candidate(**updates: object) -> GeneratedCheckCandidate:
    payload = {
        "proposal": {
            "proposal_id": "proposal.synthetic_quality_rule",
            "title": "Synthetic quality rule",
            "layer": "data_class",
            "uncovered_requirement": "Detect a controlled synthetic corruption absent from the catalog.",
            "why_existing_checks_are_insufficient": "The synthetic type deliberately has no registered adapter.",
            "required_independent_evidence": ["evaluator fixture"],
            "mutation_cases": ["remove the required marker"],
        },
        "implementation_path": "project/benchmarks/evaluation_check_candidates/synthetic/check.py",
        "interface_version": "evaluation_check_candidate.v1",
        "independent_oracle_path": "project/benchmarks/fixtures/synthetic.json",
        "positive_test_ids": ["positive.valid"],
        "negative_test_ids": ["negative.missing"],
        "mutation_test_ids": ["mutation.remove_marker"],
        "sandbox_verified": True,
        "provider_network_disabled": True,
        "secret_access_disabled": True,
        "replacement_check_ids": [],
        "proposed_catalog_version": "evaluation_check_catalog.v2",
    }
    payload.update(updates)
    return GeneratedCheckCandidate.model_validate(payload)


class EvaluationCheckPromotionTests(unittest.TestCase):
    def test_complete_candidate_passes_gates_without_catalog_mutation(self) -> None:
        assessment = assess_generated_check_candidate(_candidate())
        self.assertEqual(assessment.status, "promotion_gates_passed")
        self.assertFalse(assessment.catalog_mutated)

    def test_missing_oracle_and_mutation_evidence_remain_quarantined(self) -> None:
        assessment = assess_generated_check_candidate(
            _candidate(independent_oracle_path=None, mutation_test_ids=[])
        )
        self.assertEqual(assessment.status, "quarantined")
        self.assertTrue(any("oracle" in item for item in assessment.failures))
        self.assertTrue(any("mutation" in item for item in assessment.failures))

    def test_candidate_cannot_replace_trusted_checks(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            _candidate(replacement_check_ids=["contract.grounding"])


if __name__ == "__main__":
    unittest.main()
