from string import Template

from datasets import Dataset


def load_tooluse_dataset(seed: int = 42) -> tuple[Dataset, None]:
    """Load and prepare the tool-use training dataset."""
    train_path = "data/tooluse_data/train_data.json"
    Dataset.from_json("data/tooluse_data/eval_data.json")  # Parsed for schema validation.
    train_dataset = Dataset.from_json(train_path)

    def format_example(example):
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
                        output_text="\n".join(example["golden_response"]),
                    ),
                }
            ],
        }

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None
