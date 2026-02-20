from string import Template

from datasets import Dataset, load_dataset

_SUPPORTED_TASKS = {"copa", "cb", "wsc"}
_TASK_LABELS = {
    "copa": ["choice1", "choice2"],
    "cb": ["entailment", "contradiction", "neutral"],
    "wsc": ["False", "True"],
}

_TEACHER_PROMPT_TEMPLATE = Template(
    """
$prompt_text

Reference for this example:
The correct label is: $correct_label

Write your own response with a clear reasoning process.
Use this exact output structure:
Reasoning: <your reasoning>
Final Label: $correct_label
""".strip()
)


def _validate_train_indices(train_dataset: Dataset, train_indices: list[int] | None) -> list[int] | None:
    if train_indices is None:
        return None
    if len(train_indices) == 0:
        raise ValueError("train_indices must be non-empty when provided.")
    if any(type(index) is not int for index in train_indices):
        raise ValueError("train_indices must contain only integers.")
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("train_indices must not contain duplicates.")
    train_size = len(train_dataset)
    if any(index < 0 or index >= train_size for index in train_indices):
        raise ValueError(f"train_indices must be within [0, {train_size - 1}].")
    return train_indices


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in _SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {sorted(_SUPPORTED_TASKS)}")
    return normalized


def _label_name(task: str, label_id: int) -> str:
    labels = _TASK_LABELS[task]
    if label_id < 0 or label_id >= len(labels):
        raise ValueError(f"Invalid label id {label_id} for task '{task}'.")
    return labels[label_id]


def _render_student_prompt(task: str, example: dict) -> str:
    if task == "copa":
        return (
            "Task: Choose the better alternative for the requested causal relation.\n"
            "Give your reasoning, then end with exactly one final line in this format:\n"
            "Final Label: choice1 OR Final Label: choice2\n\n"
            f"Premise: {example['premise']}\n"
            f"Question type: {example['question']}\n"
            f"choice1: {example['choice1']}\n"
            f"choice2: {example['choice2']}"
        )

    if task == "cb":
        return (
            "Task: Natural language inference.\n"
            "Decide whether the hypothesis is entailed by the premise, contradicts it, or is neutral.\n"
            "Give your reasoning, then end with exactly one final line in this format:\n"
            "Final Label: entailment OR contradiction OR neutral\n\n"
            f"Premise: {example['premise']}\n"
            f"Hypothesis: {example['hypothesis']}"
        )

    if task == "wsc":
        return (
            "Task: Coreference resolution.\n"
            "Decide whether span2 refers to span1.\n"
            "Give your reasoning, then end with exactly one final line in this format:\n"
            "Final Label: True OR Final Label: False\n\n"
            f"Text: {example['text']}\n"
            f"span1: {example['span1_text']}\n"
            f"span2: {example['span2_text']}"
        )

    raise ValueError(f"Unsupported task '{task}'.")


def _render_gold_completion(task: str, example: dict, label_name: str) -> str:
    _ = task, example
    return f"Final Label: {label_name}"


def _build_teacher_prompt(prompt_text: str, correct_label: str) -> str:
    return _TEACHER_PROMPT_TEMPLATE.substitute(prompt_text=prompt_text, correct_label=correct_label)


def load_superglue_small_dataset(
    task: str,
    seed: int = 42,
    train_indices: list[int] | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and format low-data SuperGLUE tasks (COPA, CB, WSC) for SDFT."""
    normalized_task = _normalize_task(task)
    raw_dataset = load_dataset("super_glue", normalized_task)
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"]
    selected_indices = _validate_train_indices(train_dataset, train_indices)
    if selected_indices is not None:
        train_dataset = train_dataset.select(selected_indices)

    def format_train_example(example: dict) -> dict:
        prompt_text = _render_student_prompt(normalized_task, example)
        label_name = _label_name(normalized_task, int(example["label"]))
        teacher_prompt_text = _build_teacher_prompt(prompt_text=prompt_text, correct_label=label_name)
        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt_text}],
        }

    def format_eval_example(example: dict) -> dict:
        formatted = format_train_example(example)
        formatted["eval_label"] = _label_name(normalized_task, int(example["label"]))
        formatted["eval_task"] = normalized_task
        return formatted

    train_dataset = train_dataset.map(format_train_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.map(format_eval_example, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset


def load_superglue_small_sft_dataset(
    task: str,
    seed: int = 42,
    train_indices: list[int] | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and format low-data SuperGLUE tasks (COPA, CB, WSC) for SFT."""
    normalized_task = _normalize_task(task)
    raw_dataset = load_dataset("super_glue", normalized_task)
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"]
    selected_indices = _validate_train_indices(train_dataset, train_indices)
    if selected_indices is not None:
        train_dataset = train_dataset.select(selected_indices)

    def format_train_example(example: dict) -> dict:
        prompt_text = _render_student_prompt(normalized_task, example)
        label_name = _label_name(normalized_task, int(example["label"]))
        completion_text = _render_gold_completion(normalized_task, example, label_name)
        teacher_prompt_text = _build_teacher_prompt(prompt_text=prompt_text, correct_label=label_name)
        return {
            "prompt": prompt_text,
            "completion": completion_text,
            "teacher_prompt": teacher_prompt_text,
        }

    def format_eval_example(example: dict) -> dict:
        formatted = format_train_example(example)
        formatted.update(
            {
                "eval_label": _label_name(normalized_task, int(example["label"])),
                "eval_task": normalized_task,
            }
        )
        return formatted

    train_dataset = train_dataset.map(format_train_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.map(format_eval_example, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset
