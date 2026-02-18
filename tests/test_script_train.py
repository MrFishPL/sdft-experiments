import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import scripts.train as train_script


class ScriptTrainTest(unittest.TestCase):
    def test_parse_args_defaults(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = train_script.parse_args()

        self.assertEqual(args.method, "sdft")
        self.assertEqual(args.task, "tooluse")
        self.assertEqual(args.eval_num_generations, 1)
        self.assertTrue(args.eval_deterministic)
        self.assertTrue(args.eval_before_train)
        self.assertTrue(args.final_eval)
        self.assertIsNone(args.log_input_examples)
        self.assertTrue(args.log_examples_eval_only)
        self.assertEqual(args.max_steps, -1)

    def test_parse_args_overrides(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--method",
                "sft",
                "--task",
                "wsc",
                "--eval_num_generations",
                "3",
                "--no-eval_deterministic",
                "--no-eval_before_train",
                "--no-final_eval",
                "--log_input_examples",
                "--no-log_examples_eval_only",
                "--max_steps",
                "5",
            ],
        ):
            args = train_script.parse_args()

        self.assertEqual(args.method, "sft")
        self.assertEqual(args.task, "wsc")
        self.assertEqual(args.eval_num_generations, 3)
        self.assertFalse(args.eval_deterministic)
        self.assertFalse(args.eval_before_train)
        self.assertFalse(args.final_eval)
        self.assertTrue(args.log_input_examples)
        self.assertFalse(args.log_examples_eval_only)
        self.assertEqual(args.max_steps, 5)

    def _base_args(self, *, method: str, task: str) -> SimpleNamespace:
        return SimpleNamespace(
            method=method,
            learning_rate=1e-5,
            num_train_epochs=2,
            num_prompts_per_batch=4,
            task=task,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=3,
            ref_model_mixup_alpha=0.2,
            output_dir="out-dir",
            model_name="dummy-model",
            seed=11,
            eval_steps=10,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            save_steps=100,
            max_prompt_length=1024,
            max_completion_length=128,
            num_generations=4,
            eval_num_generations=1,
            warmup_steps=25,
            max_steps=-1,
            run_name="run-name",
            use_vllm=True,
            eval_deterministic=True,
            eval_before_train=True,
            final_eval=True,
            log_input_examples=None,
            log_examples_eval_only=True,
            paper_hparams=True,
        )

    def test_main_dispatches_sdft_and_runs_eval_before_after_train(self):
        parsed = self._base_args(method="sdft", task="copa")
        student_model = object()
        teacher_model = object()
        tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>")
        train_dataset = [{"prompt": "p", "teacher_prompt": "t"}]
        eval_dataset = [{"prompt": "p", "teacher_prompt": "t", "eval_label": "choice1", "eval_task": "copa"}]

        class DummyDistilTrainer:
            init_kwargs = None
            evaluate_calls = 0
            trained = False

            def __init__(self, **kwargs):
                DummyDistilTrainer.init_kwargs = kwargs

            def evaluate(self):
                DummyDistilTrainer.evaluate_calls += 1

            def train(self):
                DummyDistilTrainer.trained = True

        with (
            patch.object(train_script, "parse_args", return_value=parsed),
            patch.object(
                train_script.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=[student_model, teacher_model],
            ) as model_mock,
            patch.object(train_script.AutoTokenizer, "from_pretrained", return_value=tokenizer),
            patch.object(train_script, "load_tooluse_dataset") as tooluse_data_mock,
            patch.object(
                train_script,
                "load_superglue_small_dataset",
                return_value=(train_dataset, eval_dataset),
            ) as superglue_data_mock,
            patch.object(train_script, "load_superglue_small_sft_dataset") as sft_data_mock,
            patch.object(train_script, "_vllm_runtime_usable", return_value=True),
            patch.object(train_script, "DistilConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            patch.object(train_script, "DistilTrainer", DummyDistilTrainer),
            patch.object(train_script, "SmallDataSFTTrainer") as sft_trainer_mock,
            patch.object(train_script, "SFTConfig"),
        ):
            train_script.main()

        self.assertEqual(model_mock.call_count, 2)
        model_mock.assert_any_call("dummy-model", torch_dtype=torch.bfloat16)
        tooluse_data_mock.assert_not_called()
        superglue_data_mock.assert_called_once_with(task="copa", seed=11)
        sft_data_mock.assert_not_called()
        sft_trainer_mock.assert_not_called()
        self.assertTrue(DummyDistilTrainer.trained)
        self.assertEqual(DummyDistilTrainer.evaluate_calls, 2)

        cfg = DummyDistilTrainer.init_kwargs["args"]
        self.assertEqual(cfg.metric_for_best_model, "eval_small_data_accuracy")
        self.assertEqual(cfg.eval_num_generations, 1)
        self.assertTrue(cfg.eval_deterministic)
        self.assertTrue(cfg.log_completions)  # default True for small-data tasks
        self.assertTrue(cfg.log_examples_eval_only)

    def test_main_dispatches_sft_and_uses_small_data_trainer(self):
        parsed = self._base_args(method="sft", task="wsc")
        student_model = object()
        tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>")
        train_dataset = [{"prompt": "p", "completion": "c", "teacher_prompt": "tp", "eval_label": "True", "eval_task": "wsc"}]
        eval_dataset = [{"prompt": "p", "completion": "c", "teacher_prompt": "tp", "eval_label": "True", "eval_task": "wsc"}]

        class DummySFTTrainer:
            init_kwargs = None
            evaluate_calls = 0
            trained = False

            def __init__(self, **kwargs):
                DummySFTTrainer.init_kwargs = kwargs

            def evaluate(self):
                DummySFTTrainer.evaluate_calls += 1

            def train(self):
                DummySFTTrainer.trained = True

        with (
            patch.object(train_script, "parse_args", return_value=parsed),
            patch.object(train_script.AutoModelForCausalLM, "from_pretrained", return_value=student_model) as model_mock,
            patch.object(train_script.AutoTokenizer, "from_pretrained", return_value=tokenizer),
            patch.object(train_script, "load_superglue_small_sft_dataset", return_value=(train_dataset, eval_dataset)) as sft_data_mock,
            patch.object(train_script, "load_tooluse_dataset") as tooluse_data_mock,
            patch.object(train_script, "load_superglue_small_dataset") as superglue_data_mock,
            patch.object(train_script, "SFTConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            patch.object(train_script, "SmallDataSFTTrainer", DummySFTTrainer),
            patch.object(train_script, "DistilTrainer") as distil_trainer_mock,
            patch.object(train_script, "DistilConfig"),
        ):
            train_script.main()

        self.assertEqual(model_mock.call_count, 1)
        sft_data_mock.assert_called_once_with(task="wsc", seed=11)
        tooluse_data_mock.assert_not_called()
        superglue_data_mock.assert_not_called()
        distil_trainer_mock.assert_not_called()
        self.assertTrue(DummySFTTrainer.trained)
        self.assertEqual(DummySFTTrainer.evaluate_calls, 2)

        kwargs = DummySFTTrainer.init_kwargs
        self.assertEqual(kwargs["raw_eval_dataset"], eval_dataset)
        self.assertTrue(kwargs["eval_deterministic"])
        self.assertTrue(kwargs["log_input_examples"])
        self.assertTrue(kwargs["log_examples_eval_only"])

    def test_main_skips_eval_hooks_when_disabled(self):
        parsed = self._base_args(method="sdft", task="tooluse")
        parsed.eval_strategy = "no"
        parsed.eval_before_train = False
        parsed.final_eval = False
        parsed.log_input_examples = False

        class DummyDistilTrainer:
            evaluate_calls = 0
            trained = False

            def __init__(self, **kwargs):
                pass

            def evaluate(self):
                DummyDistilTrainer.evaluate_calls += 1

            def train(self):
                DummyDistilTrainer.trained = True

        with (
            patch.object(train_script, "parse_args", return_value=parsed),
            patch.object(train_script.AutoModelForCausalLM, "from_pretrained", side_effect=[object(), object()]),
            patch.object(
                train_script.AutoTokenizer,
                "from_pretrained",
                return_value=SimpleNamespace(pad_token=None, eos_token="</s>"),
            ),
            patch.object(train_script, "load_tooluse_dataset", return_value=([{"prompt": "p", "teacher_prompt": "t"}], [])),
            patch.object(train_script, "_vllm_runtime_usable", return_value=True),
            patch.object(train_script, "DistilConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            patch.object(train_script, "DistilTrainer", DummyDistilTrainer),
        ):
            train_script.main()

        self.assertEqual(DummyDistilTrainer.evaluate_calls, 0)
        self.assertTrue(DummyDistilTrainer.trained)

    def test_main_rejects_tooluse_for_sft(self):
        parsed = self._base_args(method="sft", task="tooluse")
        with patch.object(train_script, "parse_args", return_value=parsed):
            with self.assertRaisesRegex(ValueError, "method=sft is only supported"):
                train_script.main()


if __name__ == "__main__":
    unittest.main()
