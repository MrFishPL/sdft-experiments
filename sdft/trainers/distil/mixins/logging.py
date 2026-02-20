from sdft.trainers.distil._imports import *


class LoggingMixin:
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if any(key.startswith("eval_") for key in logs) else ("train" if self.model.training else "eval")
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            eval_prefix = getattr(self, "_active_metric_key_prefix", "eval")
            metrics = {f"{eval_prefix}_{key}": val for key, val in metrics.items()}

        # Update in-place so caller-visible metrics (e.g. from Trainer.evaluate) include custom entries.
        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

        should_log_examples = self.accelerator.is_main_process and self.log_completions
        if should_log_examples and getattr(self, "log_examples_eval_only", True) and mode != "eval":
            should_log_examples = False

        if should_log_examples:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "student_input": self._logs.get("student_input", self._logs["prompt"]),
                    "teacher_input": self._logs.get("teacher_input", self._logs["prompt"]),
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
