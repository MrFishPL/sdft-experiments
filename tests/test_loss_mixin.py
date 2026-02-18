import unittest
from collections import defaultdict
from types import SimpleNamespace

import torch

from sdft.trainers.distil.mixins.loss import LossMixin


class _DummyAccelerator:
    def gather(self, tensor):
        return tensor


class _DummyLossTrainer(LossMixin):
    def __init__(self, num_loss_tokens_to_skip=0):
        self.num_loss_tokens_to_skip = num_loss_tokens_to_skip
        self.top_entropy_quantile = 1.0
        self.beta = 0.0
        self.alpha = 0
        self.use_vllm = False
        self.vllm_importance_sampling_correction = False
        self.generate_from_teacher = False
        self.current_gradient_accumulation_steps = 1
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.model = SimpleNamespace(training=True)
        self.ref_model = object()
        self.accelerator = _DummyAccelerator()

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        **kwargs,
    ):
        batch = input_ids.shape[0]

        teacher_base = torch.log_softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0)
        student_bad = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)

        all_logps = teacher_base.repeat(batch, logits_to_keep, 1)
        if model is not self.ref_model:
            all_logps = all_logps.clone()
            all_logps[:, 0, :] = student_bad  # mismatch only on first completion token

        selected_logps = torch.zeros(batch, logits_to_keep, dtype=torch.float32)
        entropies = torch.ones(batch, logits_to_keep, dtype=torch.float32) * 0.5
        return selected_logps, all_logps, entropies


class LossMixinTest(unittest.TestCase):
    def _make_inputs(self):
        prompt_ids = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt_ids)
        completion_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        completion_mask = torch.ones_like(completion_ids)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": prompt_ids,
            "teacher_prompt_mask": prompt_mask,
        }

    def test_compute_loss_rejects_return_outputs(self):
        trainer = _DummyLossTrainer()
        with self.assertRaisesRegex(ValueError, "does not support returning outputs"):
            trainer.compute_loss(model=None, inputs=self._make_inputs(), return_outputs=True)

    def test_token_skip_reduces_loss_when_mismatch_is_at_start(self):
        inputs = self._make_inputs()

        no_skip = _DummyLossTrainer(num_loss_tokens_to_skip=0)
        skip_first = _DummyLossTrainer(num_loss_tokens_to_skip=1)

        loss_no_skip = no_skip._compute_loss(model=None, inputs=inputs)
        loss_skip = skip_first._compute_loss(model=None, inputs=inputs)

        self.assertGreater(loss_no_skip.item(), 0.0)
        self.assertLess(loss_skip.item(), loss_no_skip.item())
        self.assertAlmostEqual(loss_skip.item(), 0.0, places=6)

    def test_metrics_are_recorded(self):
        trainer = _DummyLossTrainer(num_loss_tokens_to_skip=0)
        _ = trainer._compute_loss(model=None, inputs=self._make_inputs())

        self.assertIn("kl_approx", trainer._metrics["train"])
        self.assertIn("entropy", trainer._metrics["train"])
        self.assertGreater(len(trainer._metrics["train"]["kl_approx"]), 0)

    def test_compute_loss_handles_missing_current_gradient_accumulation_steps(self):
        trainer = _DummyLossTrainer(num_loss_tokens_to_skip=0)
        del trainer.current_gradient_accumulation_steps
        trainer.model.training = False

        loss = trainer._compute_loss(model=None, inputs=self._make_inputs())

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("kl_approx", trainer._metrics["eval"])
        self.assertIn("entropy", trainer._metrics["eval"])


if __name__ == "__main__":
    unittest.main()
