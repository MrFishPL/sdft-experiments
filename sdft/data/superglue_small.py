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

This is an example answer with reasoning:
$output_text

Now answer with a response of your own.
Remember: your last line must be exactly `Final Label: <label>`.
""".strip()
)


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


def _render_demonstration(task: str, example: dict, label_name: str) -> str:
    if task == "copa":
        if label_name == "choice1":
            answer_text = example["choice1"]
        else:
            answer_text = example["choice2"]
        return (
            "Reasoning: Compare both options with the requested causal relation and pick the one that best fits.\n"
            f"Reasoning: The selected option is more plausible here: {answer_text}\n"
            f"Final Label: {label_name}"
        )

    if task == "cb":
        return (
            "Reasoning: Check whether the hypothesis is fully supported, contradicted, or under-specified by the premise.\n"
            f"Reasoning: The correct relation is {label_name}.\n"
            f"Final Label: {label_name}"
        )

    if task == "wsc":
        relation = "does" if label_name == "True" else "does not"
        return (
            "Reasoning: Resolve whether span2 refers back to span1 in context.\n"
            f"Reasoning: In this sentence, span2 {relation} refer to span1.\n"
            f"Final Label: {label_name}"
        )

    raise ValueError(f"Unsupported task '{task}'.")


def load_superglue_small_dataset(task: str, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Load and format low-data SuperGLUE tasks (COPA, CB, WSC) for SDFT."""
    normalized_task = _normalize_task(task)
    raw_dataset = load_dataset("super_glue", normalized_task)
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"]

    def format_train_example(example: dict) -> dict:
        prompt_text = _render_student_prompt(normalized_task, example)
        label_name = _label_name(normalized_task, int(example["label"]))
        teacher_prompt_text = _TEACHER_PROMPT_TEMPLATE.substitute(
            prompt_text=prompt_text,
            output_text=_render_demonstration(normalized_task, example, label_name),
        )
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
