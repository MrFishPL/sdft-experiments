import unittest
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


if __name__ == "__main__":
    unittest.main()
