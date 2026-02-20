#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-wsc}" # tooluse|copa|cb|wsc
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
DEVICE_BS="${DEVICE_BS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-8}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-8}"
EVAL_STEPS="${EVAL_STEPS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10}"
SEED="${SEED:-42}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/runs}"
TARGET_UPDATES="${TARGET_UPDATES:-512}"
NUM_GENERATIONS="${NUM_GENERATIONS:-1}"
FEWSHOT_INDICES_FILE="${FEWSHOT_INDICES_FILE:-${REPO_ROOT}/data/superglue_fewshot_5shot_curated.json}"
FEWSHOT_NUM_EXAMPLES="${FEWSHOT_NUM_EXAMPLES:--1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [[ -z "${MAX_COMPLETION_LENGTH}" ]]; then
  if [[ "${TASK}" == "tooluse" ]]; then
    MAX_COMPLETION_LENGTH=2048
  else
    MAX_COMPLETION_LENGTH=128
  fi
fi

resolve_fewshot_train_samples() {
  if [[ "${TASK}" == "tooluse" ]]; then
    uv run python - <<'PY'
from sdft.data import load_tooluse_one_per_name_indices
print(len(load_tooluse_one_per_name_indices()))
PY
  else
    if [[ ! -f "${FEWSHOT_INDICES_FILE}" ]]; then
      echo "Few-shot index file not found: ${FEWSHOT_INDICES_FILE}" >&2
      exit 1
    fi
    uv run python - "${TASK}" "${FEWSHOT_INDICES_FILE}" <<'PY'
import json
import sys
task = sys.argv[1]
path = sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
tasks_payload = payload.get("tasks", {})
task_payload = tasks_payload.get(task, {})
indices = task_payload.get("train_indices")
if not isinstance(indices, list):
    raise SystemExit(f"Missing train_indices for task '{task}' in {path}")
print(len(indices))
PY
  fi
}

TRAIN_SAMPLES="$(resolve_fewshot_train_samples)"
if (( FEWSHOT_NUM_EXAMPLES > 0 )) && (( FEWSHOT_NUM_EXAMPLES != TRAIN_SAMPLES )); then
  echo "FEWSHOT_NUM_EXAMPLES=${FEWSHOT_NUM_EXAMPLES} does not match selected samples ${TRAIN_SAMPLES}." >&2
  exit 1
fi
if (( FEWSHOT_NUM_EXAMPLES <= 0 )); then
  FEWSHOT_NUM_EXAMPLES="${TRAIN_SAMPLES}"
fi

EFFECTIVE_BS=$(( DEVICE_BS * ACCUM_STEPS * NUM_GPUS ))
if (( EFFECTIVE_BS <= 0 )); then
  echo "Invalid effective batch size: ${EFFECTIVE_BS}" >&2
  exit 1
fi
if (( TARGET_UPDATES <= 0 )); then
  echo "TARGET_UPDATES must be >= 1 (got ${TARGET_UPDATES})" >&2
  exit 1
fi
if (( NUM_GENERATIONS < 1 )); then
  echo "NUM_GENERATIONS must be >= 1 (got ${NUM_GENERATIONS})" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${TASK}_sft_fewshot_ng${NUM_GENERATIONS}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}_${TIMESTAMP}"
LOG_FILE="${LOG_ROOT}/${RUN_NAME}_${TIMESTAMP}.log"

CMD=(
  uv run accelerate launch --num_processes "${NUM_GPUS}" -m scripts.train
  --method sft
  --task "${TASK}"
  --output_dir "${RUN_DIR}"
  --model_name "${MODEL_NAME}"
  --seed "${SEED}"
  --learning_rate "${LEARNING_RATE}"
  --target_updates "${TARGET_UPDATES}"
  --per_device_train_batch_size "${DEVICE_BS}"
  --gradient_accumulation_steps "${ACCUM_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --eval_strategy steps
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}"
  --save_steps 100
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --num_generations "${NUM_GENERATIONS}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --warmup_steps "${WARMUP_STEPS}"
  --eval_deterministic
  --eval_before_train
  --paper_hparams
  --run_name "${RUN_NAME}"
  --log_input_examples
  --log_examples_eval_only
  --fewshot_num_examples "${FEWSHOT_NUM_EXAMPLES}"
)

if [[ "${TASK}" == "tooluse" ]]; then
  CMD+=(--tooluse_fewshot_one_per_name)
else
  CMD+=(--fewshot_indices_file "${FEWSHOT_INDICES_FILE}")
fi

echo "=== SFT FEWSHOT ==="
echo "task=${TASK}"
echo "num_generations=${NUM_GENERATIONS}"
echo "train_samples=${TRAIN_SAMPLES}"
echo "effective_bs=${EFFECTIVE_BS}"
echo "target_updates=${TARGET_UPDATES}"
echo "run_dir=${RUN_DIR}"
echo "log_file=${LOG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command: %q ' "${CMD[@]}"
  echo ""
  exit 0
fi

HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONUNBUFFERED=1 \
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
