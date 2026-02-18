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

        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.num_train_epochs, 2)
        self.assertEqual(args.num_prompts_per_batch, 32)
        self.assertEqual(args.per_device_train_batch_size, 1)
        self.assertIsNone(args.gradient_accumulation_steps)
        self.assertEqual(args.ref_model_mixup_alpha, 0.02)
        self.assertEqual(args.eval_steps, 100)
        self.assertEqual(args.eval_strategy, "steps")
        self.assertEqual(args.per_device_eval_batch_size, 8)
        self.assertEqual(args.max_completion_length, 2048)
        self.assertEqual(args.num_generations, 1)
        self.assertEqual(args.warmup_steps, 10)
        self.assertTrue(args.use_vllm)
        self.assertTrue(args.paper_hparams)
        self.assertEqual(args.model_name, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(args.seed, 42)

    def test_parse_args_overrides(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--learning_rate",
                "1e-4",
                "--num_train_epochs",
                "3",
                "--num_prompts_per_batch",
                "8",
                "--per_device_train_batch_size",
                "2",
                "--gradient_accumulation_steps",
                "6",
                "--model_name",
                "tiny-model",
                "--seed",
                "7",
                "--eval_steps",
                "25",
                "--eval_strategy",
                "epoch",
                "--num_generations",
                "1",
                "--warmup_steps",
                "42",
                "--no-paper_hparams",
            ],
        ):
            args = train_script.parse_args()

        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.num_train_epochs, 3)
        self.assertEqual(args.num_prompts_per_batch, 8)
        self.assertEqual(args.per_device_train_batch_size, 2)
        self.assertEqual(args.gradient_accumulation_steps, 6)
        self.assertEqual(args.model_name, "tiny-model")
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.eval_steps, 25)
        self.assertEqual(args.eval_strategy, "epoch")
        self.assertEqual(args.num_generations, 1)
        self.assertEqual(args.warmup_steps, 42)
        self.assertFalse(args.paper_hparams)

    def test_main_wires_components_and_triggers_train(self):
        parsed = SimpleNamespace(
            learning_rate=1e-5,
            num_train_epochs=2,
            num_prompts_per_batch=4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=3,
            ref_model_mixup_alpha=0.2,
            output_dir="out-dir",
            model_name="dummy-model",
            seed=11,
            eval_steps=100,
            eval_strategy="steps",
            per_device_eval_batch_size=1,
            save_steps=100,
            max_prompt_length=1024,
            max_completion_length=2048,
            num_generations=1,
            warmup_steps=25,
            run_name="run-name",
            use_vllm=True,
            paper_hparams=True,
        )

        student_model = object()
        teacher_model = object()
        tokenizer = object()
        train_dataset = [{"prompt": "p", "teacher_prompt": "t"}]
        eval_dataset = [{"prompt": "p", "teacher_prompt": "t", "golden_answer": []}]

        class DummyTrainer:
            init_kwargs = None
            trained = False

            def __init__(self, **kwargs):
                DummyTrainer.init_kwargs = kwargs

            def train(self):
                DummyTrainer.trained = True

        with (
            patch.object(train_script, "parse_args", return_value=parsed),
            patch.object(train_script.AutoModelForCausalLM, "from_pretrained", side_effect=[student_model, teacher_model]) as model_mock,
            patch.object(train_script.AutoTokenizer, "from_pretrained", return_value=tokenizer) as tokenizer_mock,
            patch.object(train_script, "load_tooluse_dataset", return_value=(train_dataset, eval_dataset)) as data_mock,
            patch.object(train_script, "_vllm_runtime_usable", return_value=True),
            patch.object(train_script, "DistilConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)) as config_mock,
            patch.object(train_script, "DistilTrainer", DummyTrainer),
        ):
            train_script.main()

        self.assertEqual(model_mock.call_count, 2)
        model_mock.assert_any_call("dummy-model", torch_dtype=torch.bfloat16)
        tokenizer_mock.assert_called_once_with("dummy-model")
        data_mock.assert_called_once_with(11)
        self.assertEqual(config_mock.call_count, 1)

        self.assertTrue(DummyTrainer.trained)
        self.assertIs(DummyTrainer.init_kwargs["model"], student_model)
        self.assertIs(DummyTrainer.init_kwargs["ref_model"], teacher_model)
        self.assertIs(DummyTrainer.init_kwargs["processing_class"], tokenizer)
        self.assertEqual(DummyTrainer.init_kwargs["train_dataset"], train_dataset)
        self.assertEqual(DummyTrainer.init_kwargs["eval_dataset"], eval_dataset)

        cfg = DummyTrainer.init_kwargs["args"]
        self.assertTrue(cfg.use_vllm)
        self.assertEqual(cfg.seed, 11)
        self.assertEqual(cfg.ref_model_mixup_alpha, 0.2)
        self.assertEqual(cfg.output_dir, "out-dir")
        self.assertEqual(cfg.eval_steps, 100)
        self.assertEqual(cfg.eval_strategy, "steps")
        self.assertEqual(cfg.num_generations, 1)
        self.assertEqual(cfg.per_device_train_batch_size, 2)
        self.assertEqual(cfg.gradient_accumulation_steps, 3)
        self.assertTrue(cfg.load_best_model_at_end)
        self.assertEqual(cfg.metric_for_best_model, "eval_tooluse_strict_match")
        self.assertTrue(cfg.greater_is_better)
        self.assertEqual(cfg.warmup_steps, 10)  # paper_hparams enforces paper default


if __name__ == "__main__":
    unittest.main()
