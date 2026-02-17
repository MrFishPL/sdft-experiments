import unittest

from scripts.sweep_tooluse import build_sweep_grid, extract_best_eval_metrics, sort_results


class SweepToolUseTest(unittest.TestCase):
    def test_build_sweep_grid_size_and_values(self):
        grid = build_sweep_grid()

        self.assertEqual(len(grid), 54)
        self.assertIn(
            {
                "learning_rate": 5e-6,
                "num_train_epochs": 1,
                "num_prompts_per_batch": 16,
                "ref_model_mixup_alpha": 0.01,
            },
            grid,
        )
        self.assertIn(
            {
                "learning_rate": 5e-5,
                "num_train_epochs": 2,
                "num_prompts_per_batch": 64,
                "ref_model_mixup_alpha": 0.05,
            },
            grid,
        )

    def test_extract_best_eval_metrics_uses_metric_then_loss(self):
        trainer_state = {
            "log_history": [
                {"step": 100, "eval_tooluse_strict_match": 0.8, "eval_loss": 0.9},
                {"step": 200, "eval_tooluse_strict_match": 0.8, "eval_loss": 0.7},
                {"step": 300, "eval_tooluse_strict_match": 0.7, "eval_loss": 0.1},
            ],
            "best_model_checkpoint": "runs/r/checkpoint-200",
        }

        metrics = extract_best_eval_metrics(trainer_state)

        self.assertEqual(metrics["eval_tooluse_strict_match"], 0.8)
        self.assertEqual(metrics["eval_loss"], 0.7)
        self.assertEqual(metrics["best_model_checkpoint"], "runs/r/checkpoint-200")

    def test_sort_results_places_best_successful_first(self):
        rows = [
            {
                "status": "ok",
                "eval_tooluse_strict_match": 0.7,
                "eval_loss": 0.2,
                "run_index": 1,
            },
            {
                "status": "ok",
                "eval_tooluse_strict_match": 0.8,
                "eval_loss": 0.4,
                "run_index": 2,
            },
            {
                "status": "ok",
                "eval_tooluse_strict_match": 0.8,
                "eval_loss": 0.3,
                "run_index": 3,
            },
            {
                "status": "failed",
                "eval_tooluse_strict_match": None,
                "eval_loss": None,
                "run_index": 4,
            },
        ]

        ranked = sort_results(rows)

        self.assertEqual(ranked[0]["run_index"], 3)
        self.assertEqual(ranked[1]["run_index"], 2)
        self.assertEqual(ranked[2]["run_index"], 1)
        self.assertEqual(ranked[3]["run_index"], 4)


if __name__ == "__main__":
    unittest.main()
