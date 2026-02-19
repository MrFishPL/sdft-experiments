from string import Template

from datasets import Dataset

_TRAIN_PATH = "data/tooluse_data/train_data.json"
_EVAL_PATH = "data/tooluse_data/eval_data.json"

_TEACHER_PROMPT_TEMPLATE = Template(
    """
$prompt_text

Reference for this example:
The correct tool call is:
$tool_call

Write your own response with a clear reasoning process, then provide the final tool call in the same Action / Action Input format.
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


def _load_raw_tooluse_splits() -> tuple[Dataset, Dataset]:
    train_dataset = Dataset.from_json(_TRAIN_PATH)
    eval_dataset = Dataset.from_json(_EVAL_PATH)
    return train_dataset, eval_dataset


def _render_tool_call(example: dict) -> str:
    golden_answer = example.get("golden_answer") or []
    lines: list[str] = []
    for step in golden_answer:
        action = str(step.get("Action", "")).strip()
        action_input = str(step.get("Action_Input", "")).strip()
        lines.append(f"Action: {action}")
        lines.append(f"Action Input: {action_input}")
    if lines:
        return "\n".join(lines)

    # Fallback for malformed rows that don't have structured tool calls.
    golden_response = example.get("golden_response") or []
    return "\n".join(str(line) for line in golden_response)


def _build_teacher_prompt(prompt_text: str, tool_call: str) -> str:
    return _TEACHER_PROMPT_TEMPLATE.substitute(prompt_text=prompt_text, tool_call=tool_call)


def load_tooluse_one_per_name_indices() -> list[int]:
    train_dataset, _ = _load_raw_tooluse_splits()
    first_index_by_name: dict[str, int] = {}
    for idx, raw_name in enumerate(train_dataset["name"]):
        name = str(raw_name).strip()
        if not name:
            continue
        if name not in first_index_by_name:
            first_index_by_name[name] = idx
    return [first_index_by_name[name] for name in sorted(first_index_by_name)]


def load_tooluse_dataset(seed: int = 42, train_indices: list[int] | None = None) -> tuple[Dataset, Dataset]:
    """Load and prepare the tool-use dataset for SDFT."""
    train_dataset, eval_dataset = _load_raw_tooluse_splits()
    selected_indices = _validate_train_indices(train_dataset, train_indices)
    if selected_indices is not None:
        train_dataset = train_dataset.select(selected_indices)

    def format_train_example(example: dict) -> dict:
        prompt_text = str(example["prompt"])
        tool_call = _render_tool_call(example)
        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": _build_teacher_prompt(prompt_text, tool_call)}],
        }

    def format_eval_example(example: dict) -> dict:
        formatted = format_train_example(example)
        formatted["golden_answer"] = example.get("golden_answer", [])
        return formatted

    train_dataset = train_dataset.map(format_train_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.map(format_eval_example, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset


def load_tooluse_sft_dataset(seed: int = 42, train_indices: list[int] | None = None) -> tuple[Dataset, Dataset]:
    """Load and prepare the tool-use dataset for classic SFT."""
    train_dataset, eval_dataset = _load_raw_tooluse_splits()
    selected_indices = _validate_train_indices(train_dataset, train_indices)
    if selected_indices is not None:
        train_dataset = train_dataset.select(selected_indices)

    def format_train_example(example: dict) -> dict:
        prompt_text = str(example["prompt"])
        tool_call = _render_tool_call(example)
        return {
            "prompt": prompt_text,
            "completion": tool_call,
            "teacher_prompt": _build_teacher_prompt(prompt_text, tool_call),
        }

    def format_eval_example(example: dict) -> dict:
        formatted = format_train_example(example)
        formatted["golden_answer"] = example.get("golden_answer", [])
        return formatted

    train_dataset = train_dataset.map(format_train_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.map(format_eval_example, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset
