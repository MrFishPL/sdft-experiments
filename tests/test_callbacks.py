import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sdft.trainers.distil.callbacks import MemoryEfficientSyncRefModelCallback


class _NoDeepSpeedState:
    deepspeed_plugin = None


class _DummyAccelerator:
    def __init__(self):
        self.unwrap_called = False

    def unwrap_model(self, model):
        self.unwrap_called = True
        return model


class MemoryEfficientCallbackTest(unittest.TestCase):
    def test_sync_param_updates_reference_in_place(self):
        model_param = nn.Parameter(torch.tensor([2.0, 4.0]))
        ref_param = nn.Parameter(torch.tensor([10.0, 20.0]))

        MemoryEfficientSyncRefModelCallback._sync_param(model_param, ref_param, alpha=0.25)
        expected = torch.tensor([8.0, 16.0])  # 0.75 * ref + 0.25 * model
        self.assertTrue(torch.allclose(ref_param.data, expected))

    def test_sync_target_model_non_zero3_path(self):
        model = nn.Linear(2, 2, bias=False)
        target = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
            target.weight.fill_(10.0)

        with patch("sdft.trainers.distil.callbacks.AcceleratorState", return_value=_NoDeepSpeedState()):
            MemoryEfficientSyncRefModelCallback.sync_target_model_memory_efficient(model, target, alpha=0.5)

        with torch.no_grad():
            self.assertTrue(torch.allclose(target.weight, torch.full_like(target.weight, 6.0)))

    def test_on_step_end_syncs_when_step_matches_interval(self):
        model = nn.Linear(2, 2, bias=False)
        ref_model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(4.0)
            ref_model.weight.fill_(0.0)

        accelerator = _DummyAccelerator()
        callback = MemoryEfficientSyncRefModelCallback(ref_model=ref_model, accelerator=accelerator)

        args = SimpleNamespace(ref_model_sync_steps=2, ref_model_mixup_alpha=0.5)
        state = SimpleNamespace(global_step=4)

        with patch("sdft.trainers.distil.callbacks.AcceleratorState", return_value=_NoDeepSpeedState()):
            callback.on_step_end(args=args, state=state, control=None, model=model)

        self.assertTrue(accelerator.unwrap_called)
        with torch.no_grad():
            self.assertTrue(torch.allclose(ref_model.weight, torch.full_like(ref_model.weight, 2.0)))

    def test_on_step_end_skips_when_step_does_not_match_interval(self):
        model = nn.Linear(2, 2, bias=False)
        ref_model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(7.0)
            ref_model.weight.fill_(1.0)

        callback = MemoryEfficientSyncRefModelCallback(ref_model=ref_model, accelerator=None)

        args = SimpleNamespace(ref_model_sync_steps=3, ref_model_mixup_alpha=0.5)
        state = SimpleNamespace(global_step=4)

        with patch("sdft.trainers.distil.callbacks.AcceleratorState", return_value=_NoDeepSpeedState()):
            callback.on_step_end(args=args, state=state, control=None, model=model)

        with torch.no_grad():
            self.assertTrue(torch.allclose(ref_model.weight, torch.full_like(ref_model.weight, 1.0)))


if __name__ == "__main__":
    unittest.main()
