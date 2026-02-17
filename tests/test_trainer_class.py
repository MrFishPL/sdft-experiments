import unittest

from sdft.trainers.distil.mixins import (
    GenerationMixin,
    LogProbsMixin,
    LoggingMixin,
    LossMixin,
    SamplingMixin,
    VLLMSyncMixin,
)
from sdft.trainers.distil.trainer import DistilTrainer


class DistilTrainerClassTest(unittest.TestCase):
    def test_class_metadata(self):
        self.assertEqual(DistilTrainer._name, "Distil")
        self.assertEqual(DistilTrainer._tag_names, ["trl", "distil"])

    def test_mro_includes_all_mixins(self):
        mro = DistilTrainer.__mro__
        self.assertIn(SamplingMixin, mro)
        self.assertIn(LogProbsMixin, mro)
        self.assertIn(VLLMSyncMixin, mro)
        self.assertIn(GenerationMixin, mro)
        self.assertIn(LossMixin, mro)
        self.assertIn(LoggingMixin, mro)


if __name__ == "__main__":
    unittest.main()
