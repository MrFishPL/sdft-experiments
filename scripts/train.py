import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdft.config import DistilConfig
from sdft.data import load_superglue_small_dataset, load_superglue_small_sft_dataset, load_tooluse_dataset
from sdft.trainers import DistilTrainer, SmallDataSFTTrainer

_SMALL_DATA_TASKS = {"copa", "cb", "wsc"}


def _vllm_runtime_usable() -> bool:
    try:
        from trl.import_utils import is_vllm_available
    except Exception:
        return False

    if not is_vllm_available():
        return False

    try:
        from trl.extras.vllm_client import VLLMClient  # noqa: F401
        from vllm import LLM, SamplingParams  # noqa: F401
    except Exception as exc:  # pragma: no cover - host/runtime specific
        print(
            f"[train.py] vLLM unavailable on this host ({exc}); falling back to use_vllm=False.",
            file=sys.stderr,
        )
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="SDFT/SFT Trainer")
    parser.add_argument(
        "--method",
        type=str,
        default="sdft",
        choices=["sdft", "sft"],
        help="Training method: self-distillation (sdft) or classic supervised fine-tuning (sft).",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument(
        "--task",
        type=str,
        default="tooluse",
        choices=["tooluse", "copa", "cb", "wsc"],
        help="Training/evaluation task.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device train micro-batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps. Overrides --num_prompts_per_batch when set.",
    )
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.02, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, default="runs/tooluse", help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--eval_steps", type=int, default=100, help="Validation cadence in steps")
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation scheduling strategy",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval micro-batch size")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint save cadence in steps")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Maximum completion length")
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help="Number of sampled completions per prompt group.",
    )
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps (overrides epochs when > 0)")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument(
        "--use_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM for generation when runtime linkage is available.",
    )
    parser.add_argument(
        "--eval_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic decoding for evaluation generations.",
    )
    parser.add_argument(
        "--eval_before_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one evaluation pass before training starts.",
    )
    parser.add_argument(
        "--final_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one evaluation pass after training completes.",
    )
    parser.add_argument(
        "--log_input_examples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log input/completion examples to W&B. Defaults to True for low-data tasks.",
    )
    parser.add_argument(
        "--log_examples_eval_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, log examples only on evaluation events.",
    )
    parser.add_argument(
        "--paper_hparams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply ToolUse paper-aligned non-swept hyperparameters.",
    )
    parser.add_argument(
        "--fewshot_indices_file",
        type=str,
        default=None,
        help="Path to JSON file with curated few-shot training indices per task.",
    )
    parser.add_argument(
        "--fewshot_num_examples",
        type=int,
        default=None,
        help="Expected number of few-shot examples selected from the index file.",
    )
    parser.add_argument(
        "--distil_generation_batch_size",
        type=int,
        default=None,
        help="Optional Distil generation_batch_size override (SDFT only).",
    )
    return parser.parse_args()


def _resolve_gradient_accumulation_steps(args: argparse.Namespace) -> int:
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else args.num_prompts_per_batch
    )
    if args.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be >= 1")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    return gradient_accumulation_steps


def _resolve_log_input_examples(args: argparse.Namespace) -> bool:
    if args.log_input_examples is not None:
        return args.log_input_examples
    return args.task in _SMALL_DATA_TASKS


def _load_fewshot_train_indices(args: argparse.Namespace) -> list[int] | None:
    if args.fewshot_indices_file is None:
        return None
    if args.task not in _SMALL_DATA_TASKS:
        raise ValueError("--fewshot_indices_file is supported only for low-data tasks: copa, cb, wsc.")

    path = Path(args.fewshot_indices_file)
    if not path.exists():
        raise ValueError(f"Few-shot index file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tasks_payload = payload.get("tasks")
    if not isinstance(tasks_payload, dict):
        raise ValueError("Few-shot index file must contain a top-level object field 'tasks'.")
    task_payload = tasks_payload.get(args.task)
    if not isinstance(task_payload, dict):
        raise ValueError(f"Few-shot index file does not contain task entry for '{args.task}'.")
    train_indices = task_payload.get("train_indices")
    if not isinstance(train_indices, list):
        raise ValueError(f"Task '{args.task}' must provide 'train_indices' as a list.")

    if args.fewshot_num_examples is not None and len(train_indices) != args.fewshot_num_examples:
        raise ValueError(
            f"Task '{args.task}' selected {len(train_indices)} examples, "
            f"but --fewshot_num_examples={args.fewshot_num_examples}."
        )
    return train_indices


def _build_distil_config(args: argparse.Namespace) -> DistilConfig:
    do_eval = args.eval_strategy != "no"
    use_vllm = args.use_vllm and _vllm_runtime_usable()
    gradient_accumulation_steps = _resolve_gradient_accumulation_steps(args)
    log_input_examples = _resolve_log_input_examples(args)

    config_kwargs = {
        "seed": args.seed,
        "use_vllm": use_vllm,
        "vllm_mode": "colocate",
        "vllm_tensor_parallel_size": 1,
        "vllm_gpu_memory_utilization": 0.3,
        "vllm_enable_sleep_mode": True,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": "cosine",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "disable_tqdm": False,
        "bf16": True,
        "fp16": False,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "eval_deterministic": args.eval_deterministic,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "max_grad_norm": 1,
        "weight_decay": 0.0,
        "optim": "adamw_torch",
        "report_to": "wandb",
        "run_name": args.run_name,
        "output_dir": args.output_dir,
        "log_completions": log_input_examples,
        "log_examples_eval_only": args.log_examples_eval_only,
        "sync_ref_model": True,
        "ref_model_sync_steps": 1,
        "ref_model_mixup_alpha": args.ref_model_mixup_alpha,
        "vllm_importance_sampling_correction": True,
        "num_loss_tokens_to_skip": 3,
        "do_eval": do_eval,
        "eval_strategy": args.eval_strategy,
    }

    if do_eval:
        metric_for_best_model = "eval_tooluse_strict_match"
        if args.task in {"copa", "cb", "wsc"}:
            metric_for_best_model = "eval_small_data_accuracy"
        config_kwargs.update(
            {
                "eval_steps": args.eval_steps,
                "save_strategy": args.eval_strategy,
                "load_best_model_at_end": True,
                "metric_for_best_model": metric_for_best_model,
                "greater_is_better": True,
            }
        )

    if args.paper_hparams:
        # Paper-aligned non-swept knobs for ToolUse runs.
        config_kwargs.update(
            {
                "lr_scheduler_type": "cosine",
                "warmup_steps": 10,
                "max_grad_norm": 1,
                "weight_decay": 0.0,
                "bf16": True,
                "optim": "adamw_torch",
            }
        )

    if args.distil_generation_batch_size is not None:
        config_kwargs["generation_batch_size"] = args.distil_generation_batch_size

    return DistilConfig(**config_kwargs)


def _build_sft_config(args: argparse.Namespace) -> SFTConfig:
    do_eval = args.eval_strategy != "no"
    gradient_accumulation_steps = _resolve_gradient_accumulation_steps(args)
    config_kwargs = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": "cosine",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "bf16": True,
        "fp16": False,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "max_grad_norm": 1,
        "weight_decay": 0.0,
        "optim": "adamw_torch",
        "report_to": "wandb",
        "run_name": args.run_name,
        "output_dir": args.output_dir,
        "do_eval": do_eval,
        "eval_strategy": args.eval_strategy,
        "packing": False,
        "completion_only_loss": True,
        "max_length": args.max_prompt_length + args.max_completion_length,
    }

    if do_eval:
        config_kwargs.update(
            {
                "eval_steps": args.eval_steps,
                "save_strategy": args.eval_strategy,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_small_data_accuracy",
                "greater_is_better": True,
            }
        )

    if args.paper_hparams:
        config_kwargs.update(
            {
                "lr_scheduler_type": "cosine",
                "warmup_steps": 10,
                "max_grad_norm": 1,
                "weight_decay": 0.0,
                "bf16": True,
                "optim": "adamw_torch",
            }
        )

    return SFTConfig(**config_kwargs)


def main() -> None:
    args = parse_args()
    if args.method == "sft" and args.task == "tooluse":
        raise ValueError("method=sft is only supported for low-data tasks: copa, cb, wsc.")
    if args.distil_generation_batch_size is not None and args.method != "sdft":
        raise ValueError("--distil_generation_batch_size is supported only when --method=sdft.")
    fewshot_train_indices = _load_fewshot_train_indices(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = None
    if args.method == "sdft":
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.method == "sft":
        train_dataset, eval_dataset = load_superglue_small_sft_dataset(
            task=args.task,
            seed=args.seed,
            train_indices=fewshot_train_indices,
        )
    else:
        if args.task == "tooluse":
            train_dataset, eval_dataset = load_tooluse_dataset(args.seed)
        else:
            train_dataset, eval_dataset = load_superglue_small_dataset(
                task=args.task,
                seed=args.seed,
                train_indices=fewshot_train_indices,
            )

    if args.method == "sft":
        config = _build_sft_config(args)
        trainer = SmallDataSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            raw_eval_dataset=eval_dataset,
            processing_class=tokenizer,
            eval_deterministic=args.eval_deterministic,
            log_input_examples=_resolve_log_input_examples(args),
            log_examples_eval_only=args.log_examples_eval_only,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
        )
    else:
        config = _build_distil_config(args)
        trainer = DistilTrainer(
            model=model,
            ref_model=teacher_model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

    if config.do_eval and args.eval_before_train and eval_dataset is not None:
        trainer.evaluate()
    trainer.train()
    if config.do_eval and args.final_eval and eval_dataset is not None:
        trainer.evaluate()


if __name__ == "__main__":
    main()
