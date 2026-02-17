import unittest

from sdft.config import DistilConfig


class DistilConfigTest(unittest.TestCase):
    def _base_kwargs(self):
        return {
            "output_dir": "tmp-test-output",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 3,
            "num_generations": 2,
            "report_to": [],
        }

    def test_defaults_compute_generation_batch_values(self):
        cfg = DistilConfig(**self._base_kwargs())
        self.assertEqual(cfg.steps_per_generation, 3)
        self.assertEqual(cfg.generation_batch_size, 6)
        self.assertTrue(cfg.bf16)

    def test_generation_batch_size_derives_steps_per_generation(self):
        cfg = DistilConfig(**self._base_kwargs(), generation_batch_size=8)
        self.assertEqual(cfg.steps_per_generation, 4)

    def test_steps_per_generation_derives_generation_batch_size(self):
        cfg = DistilConfig(**self._base_kwargs(), steps_per_generation=5)
        self.assertEqual(cfg.generation_batch_size, 10)

    def test_rejects_both_generation_batch_size_and_steps_per_generation(self):
        with self.assertRaisesRegex(ValueError, "can not be both configured"):
            DistilConfig(**self._base_kwargs(), generation_batch_size=8, steps_per_generation=4)

    def test_rejects_generation_batch_not_divisible_by_global_batch(self):
        with self.assertRaisesRegex(ValueError, "must be divisible by the global batch size"):
            DistilConfig(**self._base_kwargs(), generation_batch_size=5)

    def test_rejects_generation_batch_not_divisible_by_num_generations(self):
        kwargs = self._base_kwargs()
        kwargs["num_generations"] = 4
        with self.assertRaisesRegex(ValueError, "must be divisible by num_generations"):
            DistilConfig(**kwargs)

    def test_scale_rewards_boolean_is_normalized(self):
        cfg_true = DistilConfig(**self._base_kwargs(), scale_rewards=True)
        cfg_false = DistilConfig(**self._base_kwargs(), scale_rewards=False)
        self.assertEqual(cfg_true.scale_rewards, "group")
        self.assertEqual(cfg_false.scale_rewards, "none")

    def test_liger_loss_conflicts_with_delta(self):
        with self.assertRaisesRegex(ValueError, "does not support two-sided GRPO loss"):
            DistilConfig(**self._base_kwargs(), delta=0.5, use_liger_loss=True)


if __name__ == "__main__":
    unittest.main()
