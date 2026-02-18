import unittest
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GenerationConfig

from sdft.trainers.distil.mixins.generation import GenerationMixin


class _DummyGenerationTrainer(GenerationMixin):
    def __init__(self, training=True):
        self.model = SimpleNamespace(training=training)
        self.args = SimpleNamespace(steps_per_generation=2)
        self.num_iterations = 1
        self._step = 0
        self._buffered_inputs = None
        self.generate_calls = 0

    def _generate_and_score_completions(self, generation_batch):
        self.generate_calls += 1
        return {"tokens": torch.arange(4, dtype=torch.long).unsqueeze(1)}


class _DummyMetricAccelerator:
    def gather(self, tensor):
        return tensor


class _DummyGenerationMetricsTrainer(GenerationMixin):
    def __init__(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.accelerator = _DummyMetricAccelerator()


class _DummyVLLMClient:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompts = kwargs["prompts"]
        n = kwargs["n"]
        prompt_ids = [[i + 1] for i in range(len(prompts))]
        completion_ids = [[100 + i] for i in range(len(prompts) * n)]
        logprobs = [[-0.1] for _ in range(len(prompts) * n)]
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs}


class _DummyEvalGenerationTrainer(GenerationMixin):
    def __init__(self):
        self.model = SimpleNamespace(training=False)
        self.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True, process_index=0)
        self.args = SimpleNamespace(
            eval_num_generations=1,
            eval_deterministic=True,
            vllm_enable_sleep_mode=False,
            generation_kwargs=None,
            sync_ref_model=True,
            ds3_gather_for_generation=True,
        )
        self.num_generations = 4
        self.temperature = 1.0
        self.top_p = 0.9
        self.top_k = None
        self.min_p = None
        self.repetition_penalty = 1.0
        self.max_completion_length = 8
        self.max_prompt_length = 16
        self.use_vllm = True
        self.vllm_mode = "server"
        self.generate_from_teacher = False
        self.state = SimpleNamespace(global_step=0)
        self._last_loaded_step = -1
        self.vllm_client = _DummyVLLMClient()
        self.processing_class = object()
        self.generation_config = GenerationConfig(do_sample=True)
        self.use_transformers_paged = False

    def _move_model_to_vllm(self):
        return None


class GenerationMixinTest(unittest.TestCase):
    def test_prepare_inputs_training_uses_buffer_and_respects_step(self):
        trainer = _DummyGenerationTrainer(training=True)

        with (
            patch("sdft.trainers.distil.mixins.generation.shuffle_sequence_dict", side_effect=lambda x: x),
            patch("sdft.trainers.distil.mixins.generation.split_pixel_values_by_grid", side_effect=lambda x: x),
            patch("sdft.trainers.distil.mixins.generation.unsplit_pixel_values_by_grid", side_effect=lambda x: x),
        ):
            first = trainer._prepare_inputs({"dummy": 1})
            second = trainer._prepare_inputs({"dummy": 1})

        self.assertEqual(trainer.generate_calls, 1)
        self.assertEqual(trainer._step, 2)
        self.assertEqual(first["tokens"].shape, (2, 1))
        self.assertEqual(second["tokens"].shape, (2, 1))
        self.assertFalse(torch.equal(first["tokens"], second["tokens"]))

    def test_prepare_inputs_eval_generates_without_buffering(self):
        trainer = _DummyGenerationTrainer(training=False)

        out = trainer._prepare_inputs({"dummy": 1})

        self.assertEqual(trainer.generate_calls, 1)
        self.assertEqual(trainer._step, 0)
        self.assertEqual(out["tokens"].shape, (4, 1))

    def test_log_tooluse_eval_metrics_adds_expected_metrics(self):
        trainer = _DummyGenerationMetricsTrainer()
        inputs = [
            {
                "golden_answer": [
                    {
                        "Action": "getSpecificVerse",
                        "Action_Input": '{"book": "John", "chapter": 3, "verse": 16}',
                    }
                ]
            }
        ]
        completions = [
            'Action: getSpecificVerse\nAction Input: {"chapter": 3, "book": "John", "verse": 16}'
        ]

        trainer._log_tooluse_eval_metrics(inputs=inputs, completions_text=completions, device=torch.device("cpu"))

        self.assertIn("tooluse_strict_match", trainer._metrics["eval"])
        self.assertIn("tooluse_parse_success", trainer._metrics["eval"])
        self.assertIn("tooluse_action_name_match", trainer._metrics["eval"])
        self.assertEqual(trainer._metrics["eval"]["tooluse_strict_match"][0], 1.0)

    def test_log_small_data_eval_metrics_adds_expected_metrics(self):
        trainer = _DummyGenerationMetricsTrainer()
        inputs = [{"eval_label": "choice1", "eval_task": "copa"}]
        completions = ["Reasoning: ...\nFinal Label: choice1"]

        trainer._log_small_data_eval_metrics(inputs=inputs, completions_text=completions, device=torch.device("cpu"))

        self.assertIn("small_data_accuracy", trainer._metrics["eval"])
        self.assertIn("small_data_parse_success", trainer._metrics["eval"])
        self.assertEqual(trainer._metrics["eval"]["small_data_accuracy"][0], 1.0)
        self.assertEqual(trainer._metrics["eval"]["small_data_parse_success"][0], 1.0)

    def test_eval_vllm_server_uses_eval_num_generations_and_deterministic_decode(self):
        trainer = _DummyEvalGenerationTrainer()
        prompts = ["p0", "p1", "p2", "p3"]

        with (
            patch("sdft.trainers.distil.mixins.generation.maybe_apply_chat_template", side_effect=lambda x, _: x),
            patch("sdft.trainers.distil.mixins.generation.gather_object", side_effect=lambda x: x),
            patch("sdft.trainers.distil.mixins.generation.broadcast_object_list", side_effect=lambda *_args, **_kwargs: None),
            patch("sdft.trainers.distil.mixins.generation.profiling_context", side_effect=lambda *_args, **_kwargs: nullcontext()),
        ):
            prompt_ids, completion_ids, _, _ = trainer._generate_single_turn(prompts=prompts, images=None)

        self.assertEqual(len(prompt_ids), 4)
        self.assertEqual(len(completion_ids), 4)
        self.assertEqual(len(trainer.vllm_client.calls), 1)
        call = trainer.vllm_client.calls[0]
        self.assertEqual(call["n"], 1)
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["top_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
