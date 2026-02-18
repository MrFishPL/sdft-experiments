import unittest

import sdft
from sdft.config import DistilConfig
from sdft.data import load_superglue_small_dataset, load_superglue_small_sft_dataset, load_tooluse_dataset
from sdft.eval import score_small_data_predictions, score_tooluse_predictions
from sdft.trainers import DistilTrainer, MemoryEfficientSyncRefModelCallback, SmallDataSFTTrainer


class PublicApiTest(unittest.TestCase):
    def test_root_package_exports(self):
        self.assertEqual(sdft.__all__, [])

    def test_subpackage_exports(self):
        from sdft.config import __all__ as config_all
        from sdft.data import __all__ as data_all
        from sdft.eval import __all__ as eval_all
        from sdft.trainers import __all__ as trainers_all

        self.assertEqual(config_all, ["DistilConfig"])
        self.assertEqual(data_all, ["load_tooluse_dataset", "load_superglue_small_dataset", "load_superglue_small_sft_dataset"])
        self.assertEqual(eval_all, ["score_tooluse_predictions", "score_small_data_predictions"])
        self.assertIn("DistilTrainer", trainers_all)
        self.assertIn("SmallDataSFTTrainer", trainers_all)
        self.assertIn("MemoryEfficientSyncRefModelCallback", trainers_all)

    def test_imported_symbols_are_resolvable(self):
        self.assertTrue(callable(load_tooluse_dataset))
        self.assertTrue(callable(load_superglue_small_dataset))
        self.assertTrue(callable(load_superglue_small_sft_dataset))
        self.assertTrue(callable(score_tooluse_predictions))
        self.assertTrue(callable(score_small_data_predictions))
        self.assertTrue(isinstance(DistilConfig, type))
        self.assertTrue(isinstance(DistilTrainer, type))
        self.assertTrue(isinstance(SmallDataSFTTrainer, type))
        self.assertTrue(isinstance(MemoryEfficientSyncRefModelCallback, type))


if __name__ == "__main__":
    unittest.main()
