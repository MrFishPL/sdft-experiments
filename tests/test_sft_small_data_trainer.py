import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sdft.trainers.sft_small_data import SmallDataSFTTrainer


class SmallDataSFTTrainerTest(unittest.TestCase):
    def test_evaluate_adds_small_data_metrics(self):
        trainer = SmallDataSFTTrainer.__new__(SmallDataSFTTrainer)
        trainer.raw_eval_dataset = [
            {
                "prompt": "Prompt",
                "teacher_prompt": "Teacher prompt",
                "eval_label": "choice1",
                "eval_task": "copa",
            }
        ]
        trainer.log_input_examples = False
        trainer.log_examples_eval_only = True
        trainer.eval_deterministic = True
        trainer.state = SimpleNamespace(global_step=7)
        trainer.args = SimpleNamespace(report_to=[])
        trainer.is_world_process_zero = lambda: True
        logged = {}
        trainer.log = lambda payload: logged.update(payload)
        trainer._generate_eval_predictions = lambda: ["Reasoning...\nFinal Label: choice1"]

        with patch("sdft.trainers.sft_small_data.SFTTrainer.evaluate", return_value={"eval_loss": 0.1}):
            metrics = SmallDataSFTTrainer.evaluate(trainer)

        self.assertEqual(metrics["eval_loss"], 0.1)
        self.assertEqual(metrics["eval_small_data_accuracy"], 1.0)
        self.assertEqual(metrics["eval_small_data_parse_success"], 1.0)
        self.assertEqual(logged["eval_small_data_accuracy"], 1.0)
        self.assertEqual(logged["eval_small_data_parse_success"], 1.0)


if __name__ == "__main__":
    unittest.main()
