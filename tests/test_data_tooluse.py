import unittest

from sdft.data import load_tooluse_dataset, load_tooluse_one_per_name_indices, load_tooluse_sft_dataset


class ToolUseDataTest(unittest.TestCase):
    def test_dataset_loads_and_has_expected_schema(self):
        dataset, eval_dataset = load_tooluse_dataset(seed=123)

        self.assertGreater(len(dataset), 0)
        self.assertGreater(len(eval_dataset), 0)

        row = dataset[0]
        self.assertEqual(set(row.keys()), {"prompt", "teacher_prompt"})

        self.assertIsInstance(row["prompt"], list)
        self.assertIsInstance(row["teacher_prompt"], list)
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["teacher_prompt"][0]["role"], "user")

        teacher_text = row["teacher_prompt"][0]["content"]
        prompt_text = row["prompt"][0]["content"]
        self.assertIn(prompt_text, teacher_text)
        self.assertIn("The correct tool call is", teacher_text)

        eval_row = eval_dataset[0]
        self.assertEqual(set(eval_row.keys()), {"prompt", "teacher_prompt", "golden_answer"})
        self.assertIsInstance(eval_row["golden_answer"], list)

    def test_shuffle_is_deterministic_for_same_seed(self):
        ds_a, eval_a = load_tooluse_dataset(seed=42)
        ds_b, eval_b = load_tooluse_dataset(seed=42)

        first_a = [row["prompt"][0]["content"] for row in ds_a.select(range(5))]
        first_b = [row["prompt"][0]["content"] for row in ds_b.select(range(5))]
        self.assertEqual(first_a, first_b)
        self.assertEqual(eval_a[0]["prompt"][0]["content"], eval_b[0]["prompt"][0]["content"])

    def test_shuffle_changes_order_with_different_seed(self):
        ds_a, _ = load_tooluse_dataset(seed=42)
        ds_b, _ = load_tooluse_dataset(seed=43)

        first_a = ds_a[0]["prompt"][0]["content"]
        first_b = ds_b[0]["prompt"][0]["content"]
        self.assertNotEqual(first_a, first_b)

    def test_sft_dataset_schema(self):
        dataset, eval_dataset = load_tooluse_sft_dataset(seed=123)
        self.assertGreater(len(dataset), 0)
        self.assertGreater(len(eval_dataset), 0)

        row = dataset[0]
        self.assertEqual(set(row.keys()), {"prompt", "completion", "teacher_prompt"})
        self.assertIn("Action:", row["completion"])
        self.assertIn("The correct tool call is", row["teacher_prompt"])

        eval_row = eval_dataset[0]
        self.assertEqual(set(eval_row.keys()), {"prompt", "completion", "teacher_prompt", "golden_answer"})
        self.assertIsInstance(eval_row["golden_answer"], list)

    def test_one_per_name_indices_are_unique_and_non_empty(self):
        indices = load_tooluse_one_per_name_indices()
        self.assertGreater(len(indices), 0)
        self.assertEqual(len(indices), len(set(indices)))

    def test_train_indices_select_subset_for_both_sdft_and_sft(self):
        selected = load_tooluse_one_per_name_indices()[:5]
        ds_sdft, _ = load_tooluse_dataset(seed=42, train_indices=selected)
        ds_sft, _ = load_tooluse_sft_dataset(seed=42, train_indices=selected)
        self.assertEqual(len(ds_sdft), len(selected))
        self.assertEqual(len(ds_sft), len(selected))


if __name__ == "__main__":
    unittest.main()
