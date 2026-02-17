import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sdft.trainers.distil.mixins.logprobs import LogProbsMixin


class _DummyAccelerator:
    is_main_process = False

    def pad_across_processes(self, tensor, dim=0, pad_index=0):
        return tensor

    def gather(self, tensor):
        return tensor


class _DummyModel(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        vocab_size = 7
        logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                logits[i, j, int((input_ids[i, j].item() + j) % vocab_size)] = 5.0
        return SimpleNamespace(logits=logits)


class _DummyLogProbsTrainer(LogProbsMixin):
    def __init__(self):
        self.accelerator = _DummyAccelerator()
        self.model_kwarg_keys = set()
        self.temperature = 1.0
        self.args = SimpleNamespace(report_to=[])


class LogProbsMixinTest(unittest.TestCase):
    def test_get_high_entropy_mask_respects_padding_mask(self):
        trainer = _DummyLogProbsTrainer()
        entropies = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.3, 0.4]], dtype=torch.float32)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64)

        selected = trainer.get_high_entropy_mask(entropies, mask, threshold=0.5)

        self.assertEqual(selected.dtype, torch.bool)
        self.assertFalse(selected[0, 2].item())
        self.assertFalse(selected[1, 1].item())
        self.assertFalse(selected[1, 2].item())
        self.assertTrue(selected[0, 1].item())
        self.assertTrue(selected[1, 0].item())

    def test_get_high_entropy_mask_all_padding_returns_all_false(self):
        trainer = _DummyLogProbsTrainer()
        entropies = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        mask = torch.zeros_like(entropies, dtype=torch.int64)
        selected = trainer.get_high_entropy_mask(entropies, mask, threshold=0.5)
        self.assertTrue(torch.equal(selected, torch.zeros_like(entropies, dtype=torch.bool)))

    def test_get_per_token_logps_and_entropies_shapes(self):
        trainer = _DummyLogProbsTrainer()
        model = _DummyModel()

        input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        selected_logps, all_logps, entropies = trainer._get_per_token_logps_and_entropies(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            batch_size=1,
            compute_entropy=True,
            compute_all_logps=True,
        )

        self.assertEqual(selected_logps.shape, (2, 2))
        self.assertEqual(all_logps.shape, (2, 2, 7))
        self.assertEqual(entropies.shape, (2, 2))

    def test_get_per_token_logps_without_all_logps(self):
        trainer = _DummyLogProbsTrainer()
        model = _DummyModel()

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        selected_logps, all_logps, entropies = trainer._get_per_token_logps_and_entropies(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            compute_entropy=False,
            compute_all_logps=False,
        )

        self.assertEqual(selected_logps.shape, (1, 2))
        self.assertIsNone(all_logps)
        self.assertIsNone(entropies)


if __name__ == "__main__":
    unittest.main()
