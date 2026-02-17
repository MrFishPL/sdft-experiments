import unittest

from sdft.data import load_tooluse_dataset


class ToolUseDataTest(unittest.TestCase):
    def test_dataset_loads_and_has_expected_schema(self):
        dataset, eval_dataset = load_tooluse_dataset(seed=123)

        self.assertIsNone(eval_dataset)
        self.assertGreater(len(dataset), 0)

        row = dataset[0]
        self.assertEqual(set(row.keys()), {"prompt", "teacher_prompt"})

        self.assertIsInstance(row["prompt"], list)
        self.assertIsInstance(row["teacher_prompt"], list)
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["teacher_prompt"][0]["role"], "user")

        teacher_text = row["teacher_prompt"][0]["content"]
        prompt_text = row["prompt"][0]["content"]
        self.assertIn(prompt_text, teacher_text)
        self.assertIn("Now answer with a response of your own", teacher_text)

    def test_shuffle_is_deterministic_for_same_seed(self):
        ds_a, _ = load_tooluse_dataset(seed=42)
        ds_b, _ = load_tooluse_dataset(seed=42)

        first_a = [row["prompt"][0]["content"] for row in ds_a.select(range(5))]
        first_b = [row["prompt"][0]["content"] for row in ds_b.select(range(5))]
        self.assertEqual(first_a, first_b)

    def test_shuffle_changes_order_with_different_seed(self):
        ds_a, _ = load_tooluse_dataset(seed=42)
        ds_b, _ = load_tooluse_dataset(seed=43)

        first_a = ds_a[0]["prompt"][0]["content"]
        first_b = ds_b[0]["prompt"][0]["content"]
        self.assertNotEqual(first_a, first_b)


if __name__ == "__main__":
    unittest.main()
