import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from sdft.data import load_superglue_small_dataset, load_superglue_small_sft_dataset


def _build_mock_superglue_dataset(task: str) -> DatasetDict:
    if task == "copa":
        train = Dataset.from_dict(
            {
                "premise": [f"premise {i}" for i in range(6)],
                "choice1": [f"c1 {i}" for i in range(6)],
                "choice2": [f"c2 {i}" for i in range(6)],
                "question": ["cause", "effect", "cause", "effect", "cause", "effect"],
                "idx": list(range(6)),
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        validation = Dataset.from_dict(
            {
                "premise": ["vp 0", "vp 1"],
                "choice1": ["vc1 0", "vc1 1"],
                "choice2": ["vc2 0", "vc2 1"],
                "question": ["cause", "effect"],
                "idx": [0, 1],
                "label": [0, 1],
            }
        )
        return DatasetDict(train=train, validation=validation)

    if task == "cb":
        train = Dataset.from_dict(
            {
                "premise": [f"premise {i}" for i in range(6)],
                "hypothesis": [f"hypothesis {i}" for i in range(6)],
                "idx": list(range(6)),
                "label": [0, 1, 2, 0, 1, 2],
            }
        )
        validation = Dataset.from_dict(
            {
                "premise": ["vp 0", "vp 1", "vp 2"],
                "hypothesis": ["vh 0", "vh 1", "vh 2"],
                "idx": [0, 1, 2],
                "label": [0, 1, 2],
            }
        )
        return DatasetDict(train=train, validation=validation)

    if task == "wsc":
        train = Dataset.from_dict(
            {
                "text": [f"text {i}" for i in range(6)],
                "span1_index": [0, 0, 0, 0, 0, 0],
                "span2_index": [1, 1, 1, 1, 1, 1],
                "span1_text": ["Alice", "Bob", "Alice", "Bob", "Alice", "Bob"],
                "span2_text": ["she", "he", "she", "he", "she", "he"],
                "idx": list(range(6)),
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        validation = Dataset.from_dict(
            {
                "text": ["vtext 0", "vtext 1"],
                "span1_index": [0, 0],
                "span2_index": [1, 1],
                "span1_text": ["Alice", "Bob"],
                "span2_text": ["she", "he"],
                "idx": [0, 1],
                "label": [0, 1],
            }
        )
        return DatasetDict(train=train, validation=validation)

    raise ValueError(task)


class SuperGlueSmallDataTest(unittest.TestCase):
    def test_dataset_loads_and_has_expected_schema(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            for task in ("copa", "cb", "wsc"):
                with self.subTest(task=task):
                    dataset, eval_dataset = load_superglue_small_dataset(task=task, seed=123)
                    self.assertGreater(len(dataset), 0)
                    self.assertGreater(len(eval_dataset), 0)

                    row = dataset[0]
                    self.assertEqual(set(row.keys()), {"prompt", "teacher_prompt"})
                    self.assertIn("Final Label:", row["prompt"][0]["content"])
                    self.assertIn("Final Label:", row["teacher_prompt"][0]["content"])

                    eval_row = eval_dataset[0]
                    self.assertEqual(set(eval_row.keys()), {"prompt", "teacher_prompt", "eval_label", "eval_task"})
                    self.assertEqual(eval_row["eval_task"], task)

    def test_shuffle_is_deterministic_for_same_seed(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            ds_a, _ = load_superglue_small_dataset(task="copa", seed=42)
            ds_b, _ = load_superglue_small_dataset(task="copa", seed=42)
            order_a = [row["prompt"][0]["content"] for row in ds_a]
            order_b = [row["prompt"][0]["content"] for row in ds_b]
            self.assertEqual(order_a, order_b)

    def test_shuffle_changes_order_with_different_seed(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            ds_a, _ = load_superglue_small_dataset(task="copa", seed=42)
            ds_b, _ = load_superglue_small_dataset(task="copa", seed=43)
            order_a = [row["prompt"][0]["content"] for row in ds_a]
            order_b = [row["prompt"][0]["content"] for row in ds_b]
            self.assertNotEqual(order_a, order_b)

    def test_eval_labels_are_canonical_for_each_task(self):
        expected = {
            "copa": {"choice1", "choice2"},
            "cb": {"entailment", "contradiction", "neutral"},
            "wsc": {"True", "False"},
        }
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            for task, label_set in expected.items():
                with self.subTest(task=task):
                    _, eval_dataset = load_superglue_small_dataset(task=task, seed=123)
                    labels = {row["eval_label"] for row in eval_dataset}
                    self.assertTrue(labels.issubset(label_set))

    def test_sft_dataset_schema(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            train_dataset, eval_dataset = load_superglue_small_sft_dataset(task="wsc", seed=42)
            self.assertGreater(len(train_dataset), 0)
            self.assertGreater(len(eval_dataset), 0)

            train_row = train_dataset[0]
            self.assertEqual(
                set(train_row.keys()),
                {"prompt", "completion", "teacher_prompt"},
            )
            self.assertIn("Final Label:", train_row["completion"])
            eval_row = eval_dataset[0]
            self.assertEqual(set(eval_row.keys()), {"prompt", "completion", "teacher_prompt", "eval_label", "eval_task"})
            self.assertEqual(eval_row["eval_task"], "wsc")

    def test_sft_shuffle_is_deterministic_for_same_seed(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            ds_a, _ = load_superglue_small_sft_dataset(task="copa", seed=42)
            ds_b, _ = load_superglue_small_sft_dataset(task="copa", seed=42)
            order_a = [row["prompt"] for row in ds_a]
            order_b = [row["prompt"] for row in ds_b]
            self.assertEqual(order_a, order_b)

    def test_train_indices_select_subset_for_both_sdft_and_sft(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            selected = [0, 3, 5]
            ds_sdft, _ = load_superglue_small_dataset(task="wsc", seed=42, train_indices=selected)
            ds_sft, _ = load_superglue_small_sft_dataset(task="wsc", seed=42, train_indices=selected)

            self.assertEqual(len(ds_sdft), 3)
            self.assertEqual(len(ds_sft), 3)

            sdft_prompts = {row["prompt"][0]["content"] for row in ds_sdft}
            sft_prompts = {row["prompt"] for row in ds_sft}
            for idx in selected:
                self.assertTrue(any(f"Text: text {idx}" in prompt for prompt in sdft_prompts))
                self.assertTrue(any(f"Text: text {idx}" in prompt for prompt in sft_prompts))

    def test_train_indices_validation_rejects_empty_list(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_superglue_small_dataset(task="wsc", seed=42, train_indices=[])

    def test_train_indices_validation_rejects_duplicates(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_superglue_small_dataset(task="wsc", seed=42, train_indices=[1, 1, 2])

    def test_train_indices_validation_rejects_out_of_range(self):
        with patch(
            "sdft.data.superglue_small.load_dataset",
            side_effect=lambda name, task: _build_mock_superglue_dataset(task),
        ):
            with self.assertRaisesRegex(ValueError, "within"):
                load_superglue_small_sft_dataset(task="wsc", seed=42, train_indices=[0, 7])


if __name__ == "__main__":
    unittest.main()
