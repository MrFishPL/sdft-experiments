from string import Template

from datasets import Dataset


def load_tooluse_dataset(seed: int = 42) -> tuple[Dataset, Dataset]:
    """Load and prepare the tool-use training dataset."""
    train_path = "data/tooluse_data/train_data.json"
    eval_path = "data/tooluse_data/eval_data.json"
    train_dataset = Dataset.from_json(train_path)
    eval_dataset = Dataset.from_json(eval_path)

    def render_demonstration(example: dict) -> str:
        golden_response = example.get("golden_response") or []
        if golden_response:
            return "\n".join(golden_response)

        golden_answer = example.get("golden_answer") or []
        lines: list[str] = []
        for step in golden_answer:
            action = step.get("Action", "")
            action_input = step.get("Action_Input", "")
            lines.append(f"Action: {action}")
            lines.append(f"Action Input: {action_input}")
        return "\n".join(lines)

    def format_example(example: dict) -> dict:
        teacher_prompt = Template(
            """
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
"""
        )

        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [
                {
                    "role": "user",
                    "content": teacher_prompt.substitute(
                        orig_content=example["prompt"],
                        output_text=render_demonstration(example),
                    ),
                }
            ],
        }

    def format_eval_example(example: dict) -> dict:
        formatted = format_example(example)
        formatted["golden_answer"] = example.get("golden_answer", [])
        return formatted

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.map(format_eval_example, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset
