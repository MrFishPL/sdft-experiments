import unittest

import sdft.trainers.distil._imports as distil_imports


class ImportGuardsTest(unittest.TestCase):
    def test_optional_vllm_symbols_are_defined(self):
        self.assertTrue(hasattr(distil_imports, "VLLMClient"))
        self.assertTrue(hasattr(distil_imports, "LLM"))
        self.assertTrue(hasattr(distil_imports, "SamplingParams"))

    def test_logger_is_available(self):
        self.assertTrue(hasattr(distil_imports, "logger"))
        self.assertIsNotNone(distil_imports.logger)


if __name__ == "__main__":
    unittest.main()
