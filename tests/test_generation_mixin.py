import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sdft.trainers.distil.mixins.generation import GenerationMixin


class _DummyGenerationTrainer(GenerationMixin):
    def __init__(self, training=True):
        self.model = SimpleNamespace(training=training)
        self.args = SimpleNamespace(steps_per_generation=2)
        self.num_iterations = 1
        self._step = 0
        self._buffered_inputs = None
        self.generate_calls = 0

    def _generate_and_score_completions(self, generation_batch):
        self.generate_calls += 1
        return {"tokens": torch.arange(4, dtype=torch.long).unsqueeze(1)}


class _DummyMetricAccelerator:
    def gather(self, tensor):
        return tensor


class _DummyGenerationMetricsTrainer(GenerationMixin):
    def __init__(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.accelerator = _DummyMetricAccelerator()


class GenerationMixinTest(unittest.TestCase):
    def test_prepare_inputs_training_uses_buffer_and_respects_step(self):
        trainer = _DummyGenerationTrainer(training=True)

        with (
            patch("sdft.trainers.distil.mixins.generation.shuffle_sequence_dict", side_effect=lambda x: x),
            patch("sdft.trainers.distil.mixins.generation.split_pixel_values_by_grid", side_effect=lambda x: x),
            patch("sdft.trainers.distil.mixins.generation.unsplit_pixel_values_by_grid", side_effect=lambda x: x),
        ):
            first = trainer._prepare_inputs({"dummy": 1})
            second = trainer._prepare_inputs({"dummy": 1})

        self.assertEqual(trainer.generate_calls, 1)
        self.assertEqual(trainer._step, 2)
        self.assertEqual(first["tokens"].shape, (2, 1))
        self.assertEqual(second["tokens"].shape, (2, 1))
        self.assertFalse(torch.equal(first["tokens"], second["tokens"]))

    def test_prepare_inputs_eval_generates_without_buffering(self):
        trainer = _DummyGenerationTrainer(training=False)

        out = trainer._prepare_inputs({"dummy": 1})

        self.assertEqual(trainer.generate_calls, 1)
        self.assertEqual(trainer._step, 0)
        self.assertEqual(out["tokens"].shape, (4, 1))

    def test_log_tooluse_eval_metrics_adds_expected_metrics(self):
        trainer = _DummyGenerationMetricsTrainer()
        inputs = [
            {
                "golden_answer": [
                    {
                        "Action": "getSpecificVerse",
                        "Action_Input": '{"book": "John", "chapter": 3, "verse": 16}',
                    }
                ]
            }
        ]
        completions = [
            'Action: getSpecificVerse\nAction Input: {"chapter": 3, "book": "John", "verse": 16}'
        ]

        trainer._log_tooluse_eval_metrics(inputs=inputs, completions_text=completions, device=torch.device("cpu"))

        self.assertIn("tooluse_strict_match", trainer._metrics["eval"])
        self.assertIn("tooluse_parse_success", trainer._metrics["eval"])
        self.assertIn("tooluse_action_name_match", trainer._metrics["eval"])
        self.assertEqual(trainer._metrics["eval"]["tooluse_strict_match"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
