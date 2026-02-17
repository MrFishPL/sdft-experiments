# Example run

```
PYTHONUNBUFFERED=1 uv run python -u -m scripts.train \
  --output_dir /workspace/sdft-experiments/runs/tooluse_paper_single_20260217_185431 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --seed 42 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --num_prompts_per_batch 32 \
  --ref_model_mixup_alpha 0.02 \
  --eval_steps 100 \
  --eval_strategy steps \
  --per_device_eval_batch_size 8 \
  --save_steps 100 \
  --max_prompt_length 1024 \
  --max_completion_length 2048 \
  --warmup_steps 10 \
  --run_name tooluse_paper_single_lr1e-5_ep2_bs32_alpha0.02_s42 \
  --paper_hparams
```