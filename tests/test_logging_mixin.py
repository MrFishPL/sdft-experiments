import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from sdft.trainers.distil.mixins.logging import LoggingMixin


class _BaseRecorder:
    def __init__(self):
        self.logged_calls = []
        self.saved_calls = []

    def log(self, logs, start_time=None):
        self.logged_calls.append((logs, start_time))

    def _save_checkpoint(self, model, trial):
        self.saved_calls.append((model, trial))


class _DummyLoggingTrainer(LoggingMixin, _BaseRecorder):
    def __init__(self, training=True):
        super().__init__()
        self.model = SimpleNamespace(training=training)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.accelerator = SimpleNamespace(is_main_process=False)
        self.log_completions = False
        self.log_examples_eval_only = True
        self.num_completions_to_print = None
        self.wandb_log_unique_prompts = False
        self.args = SimpleNamespace(report_to=[], hub_model_id=None, output_dir="runs/my-model")
        self.state = SimpleNamespace(global_step=7)
        self._logs = {
            "images": [],
            "prompt": [],
            "student_input": [],
            "teacher_input": [],
            "completion": [],
            "rewards": defaultdict(list),
            "advantages": [],
        }
        self.model_card_names = []

    def create_model_card(self, model_name):
        self.model_card_names.append(model_name)


class LoggingMixinTest(unittest.TestCase):
    def test_log_merges_train_metrics_and_clears_buffer(self):
        trainer = _DummyLoggingTrainer(training=True)
        trainer._metrics["train"]["foo"] = [1.0, 3.0]

        trainer.log({"loss": 2.0}, start_time=1.23)

        logs, start_time = trainer.logged_calls[-1]
        self.assertEqual(start_time, 1.23)
        self.assertEqual(logs["loss"], 2.0)
        self.assertEqual(logs["foo"], 2.0)
        self.assertEqual(len(trainer._metrics["train"]), 0)

    def test_log_prefixes_eval_metrics(self):
        trainer = _DummyLoggingTrainer(training=False)
        trainer._metrics["eval"]["bar"] = [2.0, 4.0]

        trainer.log({"eval_loss": 1.0})

        logs, _ = trainer.logged_calls[-1]
        self.assertEqual(logs["eval_loss"], 1.0)
        self.assertEqual(logs["eval_bar"], 3.0)
        self.assertEqual(len(trainer._metrics["eval"]), 0)

    def test_log_infers_eval_mode_from_keys_even_if_model_is_training(self):
        trainer = _DummyLoggingTrainer(training=True)
        trainer._metrics["eval"]["tooluse_strict_match"] = [0.25, 0.75]

        trainer.log({"eval_loss": 1.0})

        logs, _ = trainer.logged_calls[-1]
        self.assertEqual(logs["eval_loss"], 1.0)
        self.assertEqual(logs["eval_tooluse_strict_match"], 0.5)
        self.assertEqual(len(trainer._metrics["eval"]), 0)

    def test_log_mutates_input_dict_for_trainer_evaluate_visibility(self):
        trainer = _DummyLoggingTrainer(training=False)
        trainer._metrics["eval"]["tooluse_strict_match"] = [1.0]
        raw_logs = {"eval_loss": 0.1}

        trainer.log(raw_logs)

        self.assertIn("eval_tooluse_strict_match", raw_logs)
        self.assertEqual(raw_logs["eval_tooluse_strict_match"], 1.0)

    def test_save_checkpoint_creates_model_card_with_output_dir_name(self):
        trainer = _DummyLoggingTrainer(training=True)
        trainer.args.hub_model_id = None

        trainer._save_checkpoint(model="model", trial="trial")

        self.assertEqual(trainer.model_card_names, ["my-model"])
        self.assertEqual(trainer.saved_calls, [("model", "trial")])

    def test_save_checkpoint_uses_hub_model_id_suffix_when_available(self):
        trainer = _DummyLoggingTrainer(training=True)
        trainer.args.hub_model_id = "org/custom-name"

        trainer._save_checkpoint(model="m", trial="t")

        self.assertEqual(trainer.model_card_names, ["custom-name"])

    def test_eval_only_example_logging_gate(self):
        trainer = _DummyLoggingTrainer(training=True)
        trainer.accelerator.is_main_process = True
        trainer.log_completions = True
        trainer.args.report_to = ["wandb"]
        trainer._logs["prompt"] = ["p"]
        trainer._logs["student_input"] = ["student p"]
        trainer._logs["teacher_input"] = ["teacher p"]
        trainer._logs["completion"] = ["c"]
        trainer._logs["advantages"] = [0.0]
        trainer._logs["rewards"]["main"] = [0.0]

        captured = []

        class _DummyWandb:
            run = object()

            @staticmethod
            def Image(image):
                return image

            @staticmethod
            def Table(dataframe):
                return dataframe

            @staticmethod
            def log(payload):
                captured.append(payload)

        with (
            patch("sdft.trainers.distil.mixins.logging.is_rich_available", return_value=False),
            patch("sdft.trainers.distil.mixins.logging.wandb", _DummyWandb),
        ):
            trainer.log({"loss": 1.0})
            self.assertEqual(len(captured), 0)

            trainer.log({"eval_loss": 1.0})
            self.assertEqual(len(captured), 1)
            table = captured[0]["completions"]
            self.assertIn("student_input", table.columns)
            self.assertIn("teacher_input", table.columns)


if __name__ == "__main__":
    unittest.main()
