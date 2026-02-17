import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdft.config import DistilConfig
from sdft.data import load_tooluse_dataset
from sdft.trainers import DistilTrainer


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
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
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
        default=8,
        help="Number of sampled completions per prompt group (paper-faithful ToolUse setting uses 1).",
    )
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument(
        "--use_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM for generation when runtime linkage is available.",
    )
    parser.add_argument(
        "--paper_hparams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply ToolUse paper-aligned non-swept hyperparameters.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> DistilConfig:
    do_eval = args.eval_strategy != "no"
    use_vllm = args.use_vllm and _vllm_runtime_usable()

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
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.num_prompts_per_batch,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "num_train_epochs": args.num_train_epochs,
        "save_steps": args.save_steps,
        "max_grad_norm": 1,
        "weight_decay": 0.0,
        "optim": "adamw_torch",
        "report_to": "wandb",
        "run_name": args.run_name,
        "output_dir": args.output_dir,
        "log_completions": False,
        "sync_ref_model": True,
        "ref_model_sync_steps": 1,
        "ref_model_mixup_alpha": args.ref_model_mixup_alpha,
        "vllm_importance_sampling_correction": True,
        "num_loss_tokens_to_skip": 3,
        "do_eval": do_eval,
        "eval_strategy": args.eval_strategy,
    }

    if do_eval:
        config_kwargs.update(
            {
                "eval_steps": args.eval_steps,
                "save_strategy": args.eval_strategy,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_tooluse_strict_match",
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

    return DistilConfig(**config_kwargs)


def main() -> None:
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, eval_dataset = load_tooluse_dataset(args.seed)

    config = _build_config(args)
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
