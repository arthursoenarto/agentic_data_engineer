from __future__ import annotations

import unittest
from typing import Any

from backend.agents.prompt_evolution import PromptEvolutionAgent, PromptMutation


class QueueLLM:
    model = "gpt-5.5"

    def __init__(self, mutation: PromptMutation) -> None:
        self.mutation = mutation
        self.request: dict[str, Any] | None = None

    def complete_json(self, **kwargs: Any) -> PromptMutation:
        self.request = kwargs
        kwargs["usage_tracker"].record_response(
            {
                "model": self.model,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 50,
                    "output_tokens_details": {"reasoning_tokens": 10},
                    "total_tokens": 150,
                },
            },
            requested_model=self.model,
        )
        return self.mutation


class PromptEvolutionAgentTests(unittest.TestCase):
    def test_mutation_records_trace_and_usage(self) -> None:
        evolved_prompt = " ".join(
            [
                "Return structured JSON for a deterministic pipeline.",
                "Validate every contract selection against the inventory.",
                "Consume the verified source fixture without network access.",
                "Publish consolidated Zarr output with workload-aligned chunks.",
            ]
            * 8
        )
        llm = QueueLLM(
            PromptMutation(
                reflection="The prior output used chunks that fragmented full-field reads.",
                hypothesis="Explicit workload-aligned chunk guidance should improve throughput.",
                expected_effect="Higher full-field samples per second.",
                risk="Larger chunks may increase memory use.",
                system_prompt=evolved_prompt,
            )
        )
        call = PromptEvolutionAgent(llm).mutate(
            experiment_id="gepa-test",
            iteration=1,
            candidate_id="p001",
            parent_candidate_id="p000",
            parent_system_prompt=evolved_prompt,
            fixed_user_prompt="Generate from {contract_json}.",
            task={"primary_objective": "consumer_samples_per_second"},
            evaluation_feedback={"feasible": True, "consumer_samples_per_second": 10.0},
            candidate_source="FILE: pipeline_impl.py\n```python\npass\n```",
            history=[],
        )

        self.assertEqual(call.candidate_id, "p001")
        self.assertEqual(call.llm_usage.call_count, 1)
        assert llm.request is not None
        self.assertIn("consumer_samples_per_second", llm.request["user_prompt"])


if __name__ == "__main__":
    unittest.main()
