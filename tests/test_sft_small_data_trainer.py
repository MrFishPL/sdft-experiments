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

    def test_log_eval_input_examples_retries_and_succeeds(self):
        trainer = SmallDataSFTTrainer.__new__(SmallDataSFTTrainer)
        trainer.log_input_examples = True
        trainer.raw_eval_dataset = [
            {
                "prompt": "Prompt",
                "teacher_prompt": "Teacher prompt",
                "eval_label": "choice1",
                "eval_task": "copa",
            }
        ]
        trainer.args = SimpleNamespace(report_to=["wandb"])
        trainer.state = SimpleNamespace(global_step=5)
        trainer.is_world_process_zero = lambda: True

        class _FlakyWandb:
            def __init__(self):
                self.run = object()
                self.log_calls = 0

            @staticmethod
            def Table(dataframe):
                return dataframe

            def log(self, payload):
                self.log_calls += 1
                if self.log_calls == 1:
                    raise RuntimeError("temporary network issue")

        stub = _FlakyWandb()
        with (
            patch("sdft.trainers.sft_small_data.wandb", stub),
            patch("sdft.trainers.sft_small_data.time.sleep", return_value=None),
        ):
            trainer._log_eval_input_examples(
                predictions=["Reasoning...\nFinal Label: choice1"],
                references=["choice1"],
                tasks=["copa"],
            )

        self.assertEqual(stub.log_calls, 2)

    def test_log_eval_input_examples_continues_after_exhausted_retries(self):
        trainer = SmallDataSFTTrainer.__new__(SmallDataSFTTrainer)
        trainer.log_input_examples = True
        trainer.raw_eval_dataset = [
            {
                "prompt": "Prompt",
                "teacher_prompt": "Teacher prompt",
                "eval_label": "choice1",
                "eval_task": "copa",
            }
        ]
        trainer.args = SimpleNamespace(report_to=["wandb"])
        trainer.state = SimpleNamespace(global_step=5)
        trainer.is_world_process_zero = lambda: True

        class _FailingWandb:
            def __init__(self):
                self.run = object()
                self.log_calls = 0

            @staticmethod
            def Table(dataframe):
                return dataframe

            def log(self, payload):
                self.log_calls += 1
                raise RuntimeError("network down")

        stub = _FailingWandb()
        with (
            patch("sdft.trainers.sft_small_data.wandb", stub),
            patch("sdft.trainers.sft_small_data.time.sleep", return_value=None),
        ):
            trainer._log_eval_input_examples(
                predictions=["Reasoning...\nFinal Label: choice1"],
                references=["choice1"],
                tasks=["copa"],
            )

        self.assertEqual(stub.log_calls, 3)

    def test_evaluate_raises_when_eval_prompt_contains_gold_completion(self):
        trainer = SmallDataSFTTrainer.__new__(SmallDataSFTTrainer)
        trainer.raw_eval_dataset = [
            {
                "prompt": "Prompt\nReasoning...\nFinal Label: choice1",
                "completion": "Reasoning...\nFinal Label: choice1",
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
        trainer.log = lambda payload: payload
        trainer._generate_eval_predictions = lambda: ["Reasoning...\nFinal Label: choice1"]

        with patch("sdft.trainers.sft_small_data.SFTTrainer.evaluate", return_value={"eval_loss": 0.1}):
            with self.assertRaisesRegex(ValueError, "Potential evaluation leakage"):
                SmallDataSFTTrainer.evaluate(trainer)

    def test_evaluate_does_not_raise_for_label_options_instruction(self):
        trainer = SmallDataSFTTrainer.__new__(SmallDataSFTTrainer)
        trainer.raw_eval_dataset = [
            {
                "prompt": (
                    "Task: Coreference resolution.\n"
                    "Give your reasoning, then end with exactly one final line in this format:\n"
                    "Final Label: True OR Final Label: False\n\n"
                    "Text: Alice thanked Bob because he helped.\n"
                    "span1: Bob\n"
                    "span2: he"
                ),
                "completion": "Final Label: True",
                "teacher_prompt": "Teacher prompt",
                "eval_label": "True",
                "eval_task": "wsc",
            }
        ]
        trainer.log_input_examples = False
        trainer.log_examples_eval_only = True
        trainer.eval_deterministic = True
        trainer.state = SimpleNamespace(global_step=7)
        trainer.args = SimpleNamespace(report_to=[])
        trainer.is_world_process_zero = lambda: True
        trainer.log = lambda payload: payload
        trainer._generate_eval_predictions = lambda: ["Reasoning...\nFinal Label: True"]

        with patch("sdft.trainers.sft_small_data.SFTTrainer.evaluate", return_value={"eval_loss": 0.1}):
            metrics = SmallDataSFTTrainer.evaluate(trainer)

        self.assertEqual(metrics["eval_small_data_accuracy"], 1.0)

    def test_evaluate_adds_tooluse_metrics(self):
        trainer = SmallDataSFTTrainer.__new__(SmallDataSFTTrainer)
        trainer.raw_eval_dataset = [
            {
                "prompt": "Prompt",
                "teacher_prompt": "Teacher prompt",
                "golden_answer": [{"Action": "search", "Action_Input": '{"query": "weather"}'}],
            }
        ]
        trainer.log_input_examples = False
        trainer.log_examples_eval_only = True
        trainer.eval_deterministic = True
        trainer.state = SimpleNamespace(global_step=9)
        trainer.args = SimpleNamespace(report_to=[])
        trainer.is_world_process_zero = lambda: True
        logged = {}
        trainer.log = lambda payload: logged.update(payload)
        trainer._generate_eval_predictions = lambda: ['Action: search\nAction Input: {"query": "weather"}']

        with patch("sdft.trainers.sft_small_data.SFTTrainer.evaluate", return_value={"eval_loss": 0.2}):
            metrics = SmallDataSFTTrainer.evaluate(trainer)

        self.assertEqual(metrics["eval_loss"], 0.2)
        self.assertEqual(metrics["eval_tooluse_strict_match"], 1.0)
        self.assertEqual(metrics["eval_tooluse_parse_success"], 1.0)
        self.assertEqual(metrics["eval_tooluse_action_name_match"], 1.0)
        self.assertEqual(logged["eval_tooluse_strict_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
