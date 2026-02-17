import unittest

import sdft
from sdft.config import DistilConfig
from sdft.data import load_tooluse_dataset
from sdft.trainers import DistilTrainer, MemoryEfficientSyncRefModelCallback


class PublicApiTest(unittest.TestCase):
    def test_root_package_exports(self):
        self.assertEqual(sdft.__all__, [])

    def test_subpackage_exports(self):
        from sdft.config import __all__ as config_all
        from sdft.data import __all__ as data_all
        from sdft.trainers import __all__ as trainers_all

        self.assertEqual(config_all, ["DistilConfig"])
        self.assertEqual(data_all, ["load_tooluse_dataset"])
        self.assertIn("DistilTrainer", trainers_all)
        self.assertIn("MemoryEfficientSyncRefModelCallback", trainers_all)

    def test_imported_symbols_are_resolvable(self):
        self.assertTrue(callable(load_tooluse_dataset))
        self.assertTrue(isinstance(DistilConfig, type))
        self.assertTrue(isinstance(DistilTrainer, type))
        self.assertTrue(isinstance(MemoryEfficientSyncRefModelCallback, type))


if __name__ == "__main__":
    unittest.main()
