import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdft.config import DistilConfig
from sdft.data import load_tooluse_dataset
from sdft.trainers import DistilTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()


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
    train_dataset, _ = load_tooluse_dataset(args.seed)

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=args.num_train_epochs,
        save_steps=100,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=args.output_dir,
        log_completions=False,  # True for debugging
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
