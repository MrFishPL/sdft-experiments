import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import scripts.train as train_script


class ScriptTrainTest(unittest.TestCase):
    def test_parse_args_defaults(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = train_script.parse_args()

        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.num_train_epochs, 1)
        self.assertEqual(args.num_prompts_per_batch, 32)
        self.assertEqual(args.ref_model_mixup_alpha, 0.01)
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
                "--model_name",
                "tiny-model",
                "--seed",
                "7",
            ],
        ):
            args = train_script.parse_args()

        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.num_train_epochs, 3)
        self.assertEqual(args.num_prompts_per_batch, 8)
        self.assertEqual(args.model_name, "tiny-model")
        self.assertEqual(args.seed, 7)

    def test_main_wires_components_and_triggers_train(self):
        parsed = SimpleNamespace(
            learning_rate=1e-5,
            num_train_epochs=2,
            num_prompts_per_batch=4,
            ref_model_mixup_alpha=0.2,
            output_dir="out-dir",
            model_name="dummy-model",
            seed=11,
        )

        student_model = object()
        teacher_model = object()
        tokenizer = object()
        train_dataset = [{"prompt": "p", "teacher_prompt": "t"}]

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
            patch.object(train_script, "load_tooluse_dataset", return_value=(train_dataset, None)) as data_mock,
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

        cfg = DummyTrainer.init_kwargs["args"]
        self.assertTrue(cfg.use_vllm)
        self.assertEqual(cfg.seed, 11)
        self.assertEqual(cfg.ref_model_mixup_alpha, 0.2)
        self.assertEqual(cfg.output_dir, "out-dir")


if __name__ == "__main__":
    unittest.main()
