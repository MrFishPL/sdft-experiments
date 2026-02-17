import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sdft.trainers.distil.mixins.vllm_sync import VLLMSyncMixin


class _DummyVLLMClient:
    def __init__(self):
        self.updates = []
        self.reset_called = False

    def update_named_param(self, name, param):
        self.updates.append((name, param.detach().clone()))

    def reset_prefix_cache(self):
        self.reset_called = True


class _DummyVLLMTrainer(VLLMSyncMixin):
    def __init__(self, generate_from_teacher=False):
        self.generate_from_teacher = generate_from_teacher
        self.model = nn.Linear(2, 2, bias=False)
        self.ref_model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
            self.ref_model.weight.fill_(3.0)

        self.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=None),
            is_main_process=True,
        )
        self.vllm_mode = "server"
        self.vllm_client = _DummyVLLMClient()
        self.is_fsdp_enabled = False
        self.args = SimpleNamespace(report_to=[])


class VLLMSyncMixinTest(unittest.TestCase):
    def test_fix_param_name_removes_known_prefixes(self):
        trainer = _DummyVLLMTrainer()
        fixed = trainer._fix_param_name_to_vllm(
            "_checkpoint_wrapped_module.layer._fsdp_wrapped_module.weight",
            extra_prefixes=["_fsdp_wrapped_module."],
        )
        self.assertEqual(fixed, "layer.weight")

    def test_move_model_to_vllm_syncs_student_weights(self):
        trainer = _DummyVLLMTrainer(generate_from_teacher=False)

        trainer._move_model_to_vllm()

        self.assertTrue(trainer.vllm_client.reset_called)
        self.assertEqual(len(trainer.vllm_client.updates), 1)
        name, value = trainer.vllm_client.updates[0]
        self.assertEqual(name, "weight")
        self.assertTrue(torch.allclose(value, torch.full_like(value, 1.0)))

    def test_move_model_to_vllm_syncs_teacher_weights_when_requested(self):
        trainer = _DummyVLLMTrainer(generate_from_teacher=True)

        trainer._move_model_to_vllm()

        self.assertEqual(len(trainer.vllm_client.updates), 1)
        _, value = trainer.vllm_client.updates[0]
        self.assertTrue(torch.allclose(value, torch.full_like(value, 3.0)))


if __name__ == "__main__":
    unittest.main()
