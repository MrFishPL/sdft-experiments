import unittest

from sdft.eval.small_data_metrics import score_small_data_predictions


class SmallDataMetricsTest(unittest.TestCase):
    def test_parses_final_label_and_scores_accuracy(self):
        scores = score_small_data_predictions(
            predictions=["Reasoning: x\nFinal Label: choice2"],
            references=["choice2"],
            tasks=["copa"],
        )
        self.assertEqual(scores["accuracy"], [1.0])
        self.assertEqual(scores["parse_success"], [1.0])

    def test_fallback_parsing_without_final_label(self):
        scores = score_small_data_predictions(
            predictions=["The relation is entailment based on the premise."],
            references=["entailment"],
            tasks=["cb"],
        )
        self.assertEqual(scores["accuracy"], [1.0])
        self.assertEqual(scores["parse_success"], [1.0])

    def test_parse_failure_sets_parse_success_to_zero(self):
        scores = score_small_data_predictions(
            predictions=["I cannot decide."],
            references=["False"],
            tasks=["wsc"],
        )
        self.assertEqual(scores["accuracy"], [0.0])
        self.assertEqual(scores["parse_success"], [0.0])

    def test_mixed_tasks_are_supported(self):
        scores = score_small_data_predictions(
            predictions=[
                "Reasoning...\nFinal Label: choice1",
                "Reasoning...\nFinal Label: neutral",
                "Reasoning...\nFinal Label: true",
            ],
            references=["choice1", "neutral", "True"],
            tasks=["copa", "cb", "wsc"],
        )
        self.assertEqual(scores["accuracy"], [1.0, 1.0, 1.0])
        self.assertEqual(scores["parse_success"], [1.0, 1.0, 1.0])

    def test_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            score_small_data_predictions(
                predictions=["Final Label: choice1"],
                references=["choice1", "choice2"],
                tasks=["copa"],
            )


if __name__ == "__main__":
    unittest.main()
