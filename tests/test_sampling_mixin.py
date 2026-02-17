import unittest
from types import SimpleNamespace

from trl.trainer.utils import RepeatSampler

from sdft.trainers.distil.mixins.sampling import SamplingMixin


class _DummyAccelerator:
    def prepare(self, dataloader):
        return dataloader


class _DummySamplingTrainer(SamplingMixin):
    def __init__(self):
        self._signature_columns = None
        self.train_dataset = list(range(16))
        self.data_collator = lambda batch: batch
        self._train_batch_size = 2
        self.num_generations = 2
        self.num_iterations = 3
        self.shuffle_dataset = True
        self.accelerator = _DummyAccelerator()
        self.args = SimpleNamespace(
            steps_per_generation=2,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_drop_last=False,
            dataloader_prefetch_factor=None,
            process_index=0,
            generation_batch_size=8,
            seed=123,
        )

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def _get_collator_with_removed_columns(self, data_collator, description=None):
        return data_collator


class SamplingMixinTest(unittest.TestCase):
    def test_signature_columns_are_initialized(self):
        trainer = _DummySamplingTrainer()
        self.assertIsNone(trainer._signature_columns)
        trainer._set_signature_columns_if_needed()
        self.assertEqual(trainer._signature_columns, ["prompt", "teacher_prompt", "image", "images"])

    def test_get_train_sampler_configuration(self):
        trainer = _DummySamplingTrainer()
        sampler = trainer._get_train_sampler()

        self.assertIsInstance(sampler, RepeatSampler)
        self.assertEqual(sampler.mini_repeat_count, 2)
        self.assertEqual(sampler.batch_size, 4)
        self.assertEqual(sampler.repeat_count, 6)
        self.assertTrue(sampler.shuffle)

    def test_get_eval_sampler_configuration(self):
        trainer = _DummySamplingTrainer()
        sampler = trainer._get_eval_sampler(eval_dataset=list(range(4)))

        self.assertIsInstance(sampler, RepeatSampler)
        self.assertEqual(sampler.mini_repeat_count, 2)
        self.assertEqual(sampler.seed, 123)

    def test_train_dataloader_uses_generation_batch_size(self):
        trainer = _DummySamplingTrainer()
        dataloader = trainer.get_train_dataloader()
        first_batch = next(iter(dataloader))
        self.assertEqual(len(first_batch), 4)  # _train_batch_size (2) * steps_per_generation (2)


if __name__ == "__main__":
    unittest.main()
