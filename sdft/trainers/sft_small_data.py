import re
import sys
import time
from typing import Any, Optional

import torch
from transformers import GenerationConfig
from trl import SFTTrainer
from trl.data_utils import maybe_apply_chat_template

from sdft.eval import score_small_data_predictions, score_tooluse_predictions

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency at runtime
    wandb = None

_TASK_LABELS = {
    "copa": ["choice1", "choice2"],
    "cb": ["entailment", "contradiction", "neutral"],
    "wsc": ["True", "False"],
}
_FINAL_LABEL_PATTERN = re.compile(r"final\s*label\s*:\s*([^\n\r]+)", re.IGNORECASE)
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9_ ]+")
_EVAL_PROMPT_LEAKAGE_MARKERS = (
    "reference for this example:",
    "the correct label is:",
    "the correct tool call is:",
)


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in _TASK_LABELS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {sorted(_TASK_LABELS)}")
    return normalized


def _normalize_label(task: str, raw_text: str) -> str | None:
    cleaned = _NON_ALNUM_PATTERN.sub(" ", raw_text.strip().lower())
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None

    for label in _TASK_LABELS[task]:
        label_lower = label.lower()
        if cleaned == label_lower:
            return label
        if cleaned.startswith(label_lower + " "):
            return label
    return None


def _parse_prediction_label(prediction: str, task: str) -> str | None:
    for candidate in reversed(_FINAL_LABEL_PATTERN.findall(prediction)):
        normalized = _normalize_label(task, candidate)
        if normalized is not None:
            return normalized
    return None


def _non_empty_stripped_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _contains_line_block(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    block_len = len(needle)
    for start in range(0, len(haystack) - block_len + 1):
        if haystack[start : start + block_len] == needle:
            return True
    return False


class SmallDataSFTTrainer(SFTTrainer):
    """SFT trainer wrapper that adds deterministic small-data evaluation metrics."""

    def __init__(
        self,
        *args: Any,
        raw_eval_dataset=None,
        eval_deterministic: bool = True,
        log_input_examples: bool = True,
        log_examples_eval_only: bool = True,
        max_prompt_length: int = 1024,
        max_completion_length: int = 128,
        **kwargs: Any,
    ):
        self.raw_eval_dataset = raw_eval_dataset
        self.eval_deterministic = eval_deterministic
        self.log_input_examples = log_input_examples
        self.log_examples_eval_only = log_examples_eval_only
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        super().__init__(*args, **kwargs)

    def _generate_eval_predictions(self) -> list[str]:
        if self.raw_eval_dataset is None or len(self.raw_eval_dataset) == 0:
            return []

        # Align eval prompt formatting with DistilTrainer: always generate from a
        # chat-templated user prompt so SFT/SDFT metrics are directly comparable.
        prompts: list[str] = []
        for row in self.raw_eval_dataset:
            raw_prompt = row["prompt"]
            conversational_prompt: list[dict[str, str]]
            if isinstance(raw_prompt, list):
                conversational_prompt = raw_prompt
            else:
                conversational_prompt = [{"role": "user", "content": str(raw_prompt)}]
            templated = maybe_apply_chat_template({"prompt": conversational_prompt}, self.processing_class)["prompt"]
            prompts.append(templated)
        batch_size = max(1, int(getattr(self.args, "per_device_eval_batch_size", 1)))
        predictions: list[str] = []

        tokenizer = self.processing_class
        model = self.model
        device = model.device

        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            was_training = model.training
            model.eval()
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    add_special_tokens=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}

                generation_kwargs = {
                    "max_new_tokens": self.max_completion_length,
                    "pad_token_id": tokenizer.pad_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "do_sample": not self.eval_deterministic,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": None,
                    "repetition_penalty": 1.0,
                }
                cache_implementation = getattr(self.args, "cache_implementation", None)
                if cache_implementation is not None:
                    generation_kwargs["cache_implementation"] = cache_implementation
                generation_config = GenerationConfig(**generation_kwargs)
                if self.eval_deterministic:
                    generation_config = GenerationConfig.from_dict(generation_config.to_dict())
                    generation_config.do_sample = False

                with torch.no_grad():
                    outputs = model.generate(**encoded, generation_config=generation_config, disable_compile=True)

                prompt_len = encoded["input_ids"].shape[1]
                completion_ids = outputs[:, prompt_len:]
                is_eos = completion_ids == tokenizer.eos_token_id
                eos_idx = torch.full(
                    (is_eos.size(0),),
                    is_eos.size(1),
                    dtype=torch.long,
                    device=completion_ids.device,
                )
                eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
                seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
                completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
                completion_ids_list = [c[m.bool()].tolist() for c, m in zip(completion_ids, completion_mask)]
                predictions.extend(tokenizer.batch_decode(completion_ids_list, skip_special_tokens=True))

            if was_training:
                model.train()
        finally:
            tokenizer.padding_side = old_padding_side

        return predictions

    def _assert_no_eval_prompt_leakage(self) -> None:
        if self.raw_eval_dataset is None:
            return

        completion_leak_indices: list[int] = []
        teacher_prompt_leak_indices: list[int] = []
        marker_leak_indices: list[int] = []
        for idx, row in enumerate(self.raw_eval_dataset):
            prompt_text = str(row.get("prompt", ""))
            prompt_lines = _non_empty_stripped_lines(prompt_text)
            normalized_prompt_lower = prompt_text.lower()

            if any(marker in normalized_prompt_lower for marker in _EVAL_PROMPT_LEAKAGE_MARKERS):
                marker_leak_indices.append(idx)

            completion_text = row.get("completion")
            if isinstance(completion_text, str):
                completion_text = completion_text.strip()
                completion_lines = _non_empty_stripped_lines(completion_text)
                if completion_lines and _contains_line_block(prompt_lines, completion_lines):
                    completion_leak_indices.append(idx)

            teacher_prompt_text = row.get("teacher_prompt")
            if isinstance(teacher_prompt_text, str):
                teacher_prompt_text = teacher_prompt_text.strip()
                if teacher_prompt_text and teacher_prompt_text == prompt_text:
                    teacher_prompt_leak_indices.append(idx)

        if completion_leak_indices or teacher_prompt_leak_indices or marker_leak_indices:
            details: list[str] = []
            if completion_leak_indices:
                details.append(f"completion line block found in prompt at indices {completion_leak_indices[:10]}")
            if teacher_prompt_leak_indices:
                details.append(f"prompt equals teacher_prompt at indices {teacher_prompt_leak_indices[:10]}")
            if marker_leak_indices:
                details.append(
                    f"teacher/reference leakage markers found in prompt at indices {marker_leak_indices[:10]}"
                )
            raise ValueError("Potential evaluation leakage detected: " + "; ".join(details))

    def _log_eval_input_examples(
        self,
        predictions: list[str],
        references: list[str],
        tasks: list[str],
    ) -> None:
        if not self.log_input_examples or self.raw_eval_dataset is None:
            return
        if not self.is_world_process_zero():
            return
        if not self.args.report_to or "wandb" not in self.args.report_to:
            return
        if wandb is None or wandb.run is None:
            return

        import pandas as pd

        rows = []
        for idx, row in enumerate(self.raw_eval_dataset):
            prediction = predictions[idx] if idx < len(predictions) else ""
            task = _normalize_task(tasks[idx]) if idx < len(tasks) else _normalize_task(str(row["eval_task"]))
            rows.append(
                {
                    "step": str(self.state.global_step),
                    "student_input": str(row["prompt"]),
                    "teacher_input": str(row.get("teacher_prompt", "")),
                    "completion": prediction,
                    "predicted_label": _parse_prediction_label(prediction, task),
                    "gold_label": references[idx] if idx < len(references) else str(row["eval_label"]),
                }
            )
        max_attempts = 3
        wait_seconds = [1, 2, 5]
        table = wandb.Table(dataframe=pd.DataFrame(rows))
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                wandb.log({"eval_input_examples": table})
                return
            except Exception as exc:  # pragma: no cover - network/runtime specific
                last_error = exc
                print(
                    f"[SmallDataSFTTrainer] Failed to log eval_input_examples to W&B "
                    f"(attempt {attempt}/{max_attempts}): {exc}",
                    file=sys.stderr,
                )
                if attempt < max_attempts:
                    run_id = getattr(wandb.run, "id", "<unknown>")
                    print(
                        f"[SmallDataSFTTrainer] Retrying with existing W&B run context (run_id={run_id}).",
                        file=sys.stderr,
                    )
                    time.sleep(wait_seconds[attempt - 1])
        print(
            "[SmallDataSFTTrainer] Continuing training without eval_input_examples "
            f"after {max_attempts} failed attempts: {last_error}",
            file=sys.stderr,
        )

    def _log_eval_input_examples_tooluse(
        self,
        predictions: list[str],
        references: list[list[dict[str, Any]]],
    ) -> None:
        if not self.log_input_examples or self.raw_eval_dataset is None:
            return
        if not self.is_world_process_zero():
            return
        if not self.args.report_to or "wandb" not in self.args.report_to:
            return
        if wandb is None or wandb.run is None:
            return

        import pandas as pd

        rows = []
        for idx, row in enumerate(self.raw_eval_dataset):
            prediction = predictions[idx] if idx < len(predictions) else ""
            reference = references[idx] if idx < len(references) else []
            rows.append(
                {
                    "step": str(self.state.global_step),
                    "student_input": str(row["prompt"]),
                    "teacher_input": str(row.get("teacher_prompt", "")),
                    "completion": prediction,
                    "gold_tool_calls": str(reference),
                }
            )
        max_attempts = 3
        wait_seconds = [1, 2, 5]
        table = wandb.Table(dataframe=pd.DataFrame(rows))
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                wandb.log({"eval_input_examples": table})
                return
            except Exception as exc:  # pragma: no cover - network/runtime specific
                last_error = exc
                print(
                    f"[SmallDataSFTTrainer] Failed to log eval_input_examples to W&B "
                    f"(attempt {attempt}/{max_attempts}): {exc}",
                    file=sys.stderr,
                )
                if attempt < max_attempts:
                    run_id = getattr(wandb.run, "id", "<unknown>")
                    print(
                        f"[SmallDataSFTTrainer] Retrying with existing W&B run context (run_id={run_id}).",
                        file=sys.stderr,
                    )
                    time.sleep(wait_seconds[attempt - 1])
        print(
            "[SmallDataSFTTrainer] Continuing training without eval_input_examples "
            f"after {max_attempts} failed attempts: {last_error}",
            file=sys.stderr,
        )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        if self.raw_eval_dataset is None or len(self.raw_eval_dataset) == 0:
            return metrics

        self._assert_no_eval_prompt_leakage()
        predictions = self._generate_eval_predictions()
        if all("eval_label" in row and "eval_task" in row for row in self.raw_eval_dataset):
            references = [str(row["eval_label"]) for row in self.raw_eval_dataset]
            tasks = [str(row["eval_task"]) for row in self.raw_eval_dataset]
            scored = score_small_data_predictions(predictions, references, tasks)
            extra_metrics = {
                f"{metric_key_prefix}_small_data_accuracy": sum(scored["accuracy"]) / len(scored["accuracy"]),
                f"{metric_key_prefix}_small_data_parse_success": sum(scored["parse_success"]) / len(scored["parse_success"]),
            }
            metrics.update(extra_metrics)
            self.log(extra_metrics)
            self._log_eval_input_examples(predictions=predictions, references=references, tasks=tasks)
            return metrics

        if all("golden_answer" in row for row in self.raw_eval_dataset):
            references = [row.get("golden_answer", []) for row in self.raw_eval_dataset]
            scored = score_tooluse_predictions(predictions, references)
            extra_metrics = {
                f"{metric_key_prefix}_tooluse_strict_match": sum(scored["strict_match"]) / len(scored["strict_match"]),
                f"{metric_key_prefix}_tooluse_parse_success": sum(scored["parse_success"]) / len(scored["parse_success"]),
                f"{metric_key_prefix}_tooluse_action_name_match": sum(scored["action_name_match"]) / len(scored["action_name_match"]),
            }
            metrics.update(extra_metrics)
            self.log(extra_metrics)
            self._log_eval_input_examples_tooluse(predictions=predictions, references=references)
            return metrics

        return metrics
