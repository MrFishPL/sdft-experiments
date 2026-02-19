# SDFT Experiments

## Objective
This project tests whether self-distillation fine-tuning (SDFT) is harder to overfit than classic supervised fine-tuning on low-data tasks.

Current focus is SuperGLUE small-data benchmarks (`copa`, `cb`, `wsc`), with overnight ablations on `wsc`.

## Current Capabilities
- Distillation training entrypoint: `scripts/train.py` with `--method sdft`
- Classic supervised baseline: `scripts/train.py` with `--method sft`
- Low-data dataset loaders for SuperGLUE small tasks
- Validation metrics: `eval_small_data_accuracy` (primary), `eval_small_data_parse_success` (diagnostic)
- Validation before training and final validation after training
- W&B example logging with both student input and teacher input

## Choices You Made
| Choice | Selected Option | Why |
|---|---|---|
| Baseline | Classic SFT on gold outputs | Direct comparison against SDFT for overfitting behavior |
| Model | `Qwen/Qwen2.5-3B-Instruct` | Main model for WSC ablation runs |
| Validation cadence | Every 10 steps | Frequent enough to track overfitting onset |
| Final validation | Enabled | Always get end-of-run checkpoint quality |
| Eval generations | Fixed to one pass over validation set | Keeps validation cost predictable and directly tied to dataset size |
| Eval decoding | Deterministic | Lower metric noise for fair comparison |
| Orchestrator policy | Fail-fast | Avoid wasting overnight GPU on broken config |
| Batch policy | Fixed conservative | Safer overnight utilization |
| W&B examples | Log student + teacher inputs on eval | Inspect prompt quality and drift |
| SFT step budget | Auto-aligned to longest SDFT run | Fair comparison on total optimizer steps |

## Dataset Details
Supported low-data tasks:
- `copa`: train 400, validation 100, test 500
- `cb`: train 250, validation 56, test 250
- `wsc`: train 554, validation 104, test 146

Important: official SuperGLUE test labels are hidden (`label=-1`), so local accuracy metrics are computed on `validation`, not `test`.

## Metric Policy
Use these for low-data runs:
- Primary: `eval_small_data_accuracy`
- Diagnostic: `eval_small_data_parse_success`

Checkpoint selection for low-data runs uses `eval_small_data_accuracy`.

## WSC Overnight Matrix
Scenarios run sequentially, one per process:
1. Classic FT (`method=sft`), 10 epochs
2. SDFT (`method=sdft`, `num_generations=1`), 10 epochs
3. SDFT (`method=sdft`, `num_generations=4`), 10 epochs
4. SDFT (`method=sdft`, `num_generations=8`), 10 epochs
5. SDFT (`method=sdft`, `num_generations=16`), 10 epochs
6. SDFT (`method=sdft`, `num_generations=32`), 10 epochs

For fairness, the SFT baseline is launched with `max_steps` equal to the longest SDFT scenario (computed from dataset size and effective batch size).  
Formula used by the script:

```bash
longest_sdft_steps_per_epoch = floor(TRAIN_SAMPLES * max_num_generations / (DEVICE_BS * ACCUM_STEPS * NUM_GPUS))
sft_max_steps = longest_sdft_steps_per_epoch * NUM_EPOCHS
```

If you set `MAX_STEPS>0`, that value overrides this auto-alignment for both SFT and SDFT scenarios.

## Run Commands
### Full 5-shot SuperGLUE sweep (baseline + SDFT 256->1)

```bash
cd /home/karp/sdft-experiments

# Optional for online logging
uv run wandb login

WANDB_MODE=online \
TASKS="copa cb wsc" \
DISTIL_NUM_GENERATIONS_LIST="256 128 64 32 16 8 4 1" \
EPOCHS_AT_MAX_GEN=5 \
FEWSHOT_NUM_EXAMPLES=5 \
EVAL_STEPS=10 \
bash scripts/run_superglue_fewshot_scenarios.sh
```

Notes:
- Runs baseline SFT first per task, then SDFT in order `256 -> 1`.
- Uses the curated 5-shot file: `data/superglue_fewshot_5shot_curated.json`.
- Validation runs every 10 steps and also once before training starts (`--eval_before_train`).
- Outputs are written to `runs/` and logs to `logs/runs/`.

### Few-shot smoke test

```bash
cd /home/karp/sdft-experiments
WANDB_MODE=offline bash scripts/run_superglue_fewshot_scenarios_smoke.sh
```

### Smoke test first
Run this before overnight jobs:

```bash
cd /home/karp/sdft-experiments
bash scripts/run_wsc_scenarios_smoke.sh
```

Default smoke settings:
- `MAX_STEPS=2`
- `NUM_EPOCHS=1`
- `EVAL_STEPS=1`
- smaller train/eval batch sizes

### Full overnight run in screen

```bash
screen -S wsc_night -L -Logfile /home/karp/sdft-experiments/logs/wsc_night_$(date +%Y%m%d_%H%M%S).log -dm bash -lc '
cd /home/karp/sdft-experiments && bash scripts/run_wsc_scenarios.sh
'
```

Attach:

```bash
screen -r wsc_night
```

### Override conservative batch defaults (optional)

```bash
cd /home/karp/sdft-experiments
NUM_GPUS=1 CUDA_DEVICES=0 DEVICE_BS=16 ACCUM_STEPS=4 PER_DEVICE_EVAL_BS=8 bash scripts/run_wsc_scenarios.sh
```

Effective batch size:

```bash
effective_batch_size = DEVICE_BS * ACCUM_STEPS * NUM_GPUS
```

## Single-run Manual Command (reference)

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
uv run accelerate launch --num_processes 1 -m scripts.train \
  --method sdft \
  --task wsc \
  --output_dir runs/wsc_manual \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --seed 42 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --num_generations 8 \
  --eval_deterministic \
  --eval_steps 10 \
  --eval_strategy steps \
  --per_device_eval_batch_size 8 \
  --max_prompt_length 1024 \
  --max_completion_length 128 \
  --eval_before_train \
  --final_eval \
  --log_input_examples \
  --log_examples_eval_only
```

## GPU Safety Checklist
- Run smoke matrix before full night run.
- Keep conservative defaults first; increase `DEVICE_BS` only after stable smoke.
- Keep deterministic eval for metric stability and memory safety.
- Watch logs for CUDA OOM or repeated parse failures.

## Troubleshooting
- `vLLM unavailable ... falling back to use_vllm=False`
- Expected on hosts missing required CUDA runtime linkage for vLLM.
- `Found GPU ... capability 12.1 ... supported (8.0)-(12.0)`
- PyTorch warning from local wheel build; verify training still proceeds.
- Non-intuitive step counts (for SDFT)
- `num_generations` and grouped generation affect optimizer-step accounting.
- Low parse success
- Check output formatting requirement: final line must be `Final Label: <label>`.

## Repo Map
- `scripts/train.py`: unified SDFT/SFT entrypoint
- `scripts/run_superglue_fewshot_scenarios.sh`: full 5-shot SuperGLUE scenario matrix
- `scripts/run_superglue_fewshot_scenarios_smoke.sh`: cheap preflight for the 5-shot matrix
- `scripts/run_wsc_scenarios.sh`: full WSC scenario matrix (fail-fast)
- `scripts/run_wsc_scenarios_smoke.sh`: cheap preflight matrix
- `sdft/data/superglue_small.py`: low-data dataset loaders
- `sdft/eval/small_data_metrics.py`: label parsing + accuracy metrics
- `sdft/trainers/distil/`: SDFT trainer and mixins
- `sdft/trainers/sft_small_data.py`: SFT wrapper with small-data eval metrics
