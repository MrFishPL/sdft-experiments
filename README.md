# SDFT Experiments

This repo runs controlled comparisons between classic supervised fine-tuning (SFT) and self-distillation fine-tuning (SDFT) on:
- SuperGLUE small tasks: `copa`, `cb`, `wsc`
- Tool-use dataset: `tooluse`

## Quick Setup
```bash
cd /home/karp/sdft-experiments
uv sync

# Optional but recommended for online tracking
uv run wandb login
```

## Canonical Scripts (Use These)
Use these four scripts for all experiments:
- `scripts/run_sft_full.sh`: SFT on full train split
- `scripts/run_sft_fewshot.sh`: SFT on curated few-shot subset
- `scripts/run_sdft_full.sh`: SDFT on full train split
- `scripts/run_sdft_fewshot.sh`: SDFT on curated few-shot subset

## Shared Comparability Mechanism (Equal Update Budget)
All scripts are step-based and use the same update budget by default:
- `TARGET_UPDATES=2000` (default for every method and scenario)

Effective batch size is still controlled by:
```text
effective_bs = DEVICE_BS * ACCUM_STEPS * NUM_GPUS
```

How `TARGET_UPDATES` is applied:
- SFT runs: optimizer steps = `TARGET_UPDATES`
- SDFT runs: optimizer steps = `ceil(TARGET_UPDATES / NUM_GENERATIONS)`
- SDFT therefore guarantees at least `TARGET_UPDATES` update units via:
  - `resolved_update_units = optimizer_steps * NUM_GENERATIONS`

Override if needed:
```bash
TARGET_UPDATES=4000 bash scripts/run_sdft_full.sh
```

## Evaluation Defaults
- Validation every `EVAL_STEPS` steps (default `10` in runner scripts)
- Validation at step 0 (`--eval_before_train`)
- Final validation after training (`--final_eval`)

Primary metrics:
- SuperGLUE small tasks: `eval_small_data_accuracy`
- Tool-use task: `eval_tooluse_strict_match`

## Few-Shot Data Sources
- SuperGLUE curated 5-shot subsets:
  - `data/superglue_fewshot_5shot_curated.json`
- Tool-use few-shot mode:
  - `--tooluse_fewshot_one_per_name` (exactly one sample per tool name)

Current number of tool-use prototypes:
```bash
uv run python - <<'PY'
from sdft.data import load_tooluse_one_per_name_indices
print(len(load_tooluse_one_per_name_indices()))
PY
```

## Example 1: Full-Data SFT Baseline (WSC)
```bash
TASK=wsc \
EVAL_STEPS=10 \
bash scripts/run_sft_full.sh
```

## Example 2: Full-Data SDFT (WSC, g=16)
```bash
TASK=wsc \
NUM_GENERATIONS=16 \
EVAL_STEPS=10 \
bash scripts/run_sdft_full.sh
```

## Example 3: Few-Shot SuperGLUE (WSC 5-shot)
SFT baseline first:
```bash
TASK=wsc \
FEWSHOT_INDICES_FILE=data/superglue_fewshot_5shot_curated.json \
FEWSHOT_NUM_EXAMPLES=5 \
EVAL_STEPS=10 \
bash scripts/run_sft_fewshot.sh
```

Then SDFT sweep:
```bash
for g in 256 128 64 32 16 8 4 1; do
  TASK=wsc \
  NUM_GENERATIONS="$g" \
  FEWSHOT_INDICES_FILE=data/superglue_fewshot_5shot_curated.json \
  FEWSHOT_NUM_EXAMPLES=5 \
  EVAL_STEPS=10 \
  bash scripts/run_sdft_fewshot.sh
done
```

## Example 4: Few-Shot ToolUse (One per tool name)
Resolve prototype count first:
```bash
TOOLUSE_N="$(uv run python - <<'PY'
from sdft.data import load_tooluse_one_per_name_indices
print(len(load_tooluse_one_per_name_indices()))
PY
)"
echo "${TOOLUSE_N}"
```

SFT few-shot baseline:
```bash
TASK=tooluse \
FEWSHOT_NUM_EXAMPLES="${TOOLUSE_N}" \
EVAL_STEPS=10 \
bash scripts/run_sft_fewshot.sh
```

SDFT few-shot:
```bash
TASK=tooluse \
NUM_GENERATIONS=4 \
FEWSHOT_NUM_EXAMPLES="${TOOLUSE_N}" \
EVAL_STEPS=10 \
bash scripts/run_sdft_fewshot.sh
```

## Example 5: Full Few-Shot SuperGLUE Sweep (All 3 tasks)
```bash
for task in copa cb wsc; do
  TASK="$task" \
  FEWSHOT_INDICES_FILE=data/superglue_fewshot_5shot_curated.json \
  FEWSHOT_NUM_EXAMPLES=5 \
  EVAL_STEPS=10 \
  bash scripts/run_sft_fewshot.sh

  for g in 256 128 64 32 16 8 4 1; do
    TASK="$task" \
    NUM_GENERATIONS="$g" \
    FEWSHOT_INDICES_FILE=data/superglue_fewshot_5shot_curated.json \
    FEWSHOT_NUM_EXAMPLES=5 \
    EVAL_STEPS=10 \
    bash scripts/run_sdft_fewshot.sh
  done
done
```

## Dry-Run / Preflight
Print resolved command without launching training:
```bash
DRY_RUN=1 TASK=wsc NUM_GENERATIONS=16 bash scripts/run_sdft_full.sh
DRY_RUN=1 TASK=wsc bash scripts/run_sft_full.sh
DRY_RUN=1 TASK=wsc NUM_GENERATIONS=16 FEWSHOT_NUM_EXAMPLES=5 FEWSHOT_INDICES_FILE=data/superglue_fewshot_5shot_curated.json bash scripts/run_sdft_fewshot.sh
DRY_RUN=1 TASK=wsc FEWSHOT_NUM_EXAMPLES=5 FEWSHOT_INDICES_FILE=data/superglue_fewshot_5shot_curated.json bash scripts/run_sft_fewshot.sh
```

## Important Runtime Knobs
Common environment variables for all scripts:
- `TASK`: `tooluse|copa|cb|wsc`
- `MODEL_NAME` (default `Qwen/Qwen2.5-3B-Instruct`)
- `NUM_GPUS` (default `1`)
- `CUDA_DEVICES` (default `0`)
- `DEVICE_BS` (default `4`)
- `ACCUM_STEPS` (default `8`)
- `PER_DEVICE_EVAL_BS` (default `8`)
- `EVAL_STEPS` (default `10`)
- `TARGET_UPDATES` (default `2000`)
- `LEARNING_RATE` (default `1e-5`)
- `MAX_GRAD_NORM` (default `10`)

SDFT-only:
- `NUM_GENERATIONS`
- `DISTIL_ALPHA` (default `1.0`)
- `REF_MODEL_MIXUP_ALPHA` (default `0.02`)

Few-shot-only:
- SuperGLUE: `FEWSHOT_INDICES_FILE`, `FEWSHOT_NUM_EXAMPLES`
- ToolUse: `FEWSHOT_NUM_EXAMPLES` + automatic `--tooluse_fewshot_one_per_name`

## Output Locations
- Run artifacts/checkpoints: `runs/`
- Script logs: `logs/runs/`

## Troubleshooting
- If W&B network/auth fails during example table logging, training now continues (best-effort logging).
- If step counts look unexpected, check your `TARGET_UPDATES`, `DEVICE_BS`, `ACCUM_STEPS`, and `NUM_GPUS`.
- If you changed subset mode (full vs few-shot), use the matching script.
