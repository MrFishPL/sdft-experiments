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
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [[ -z "${MAX_COMPLETION_LENGTH}" ]]; then
  if [[ "${TASK}" == "tooluse" ]]; then
    MAX_COMPLETION_LENGTH=2048
  else
    MAX_COMPLETION_LENGTH=128
  fi
fi

resolve_full_train_samples() {
  case "${TASK}" in
    copa) echo 400 ;;
    cb) echo 250 ;;
    wsc) echo 554 ;;
    tooluse)
      uv run python - <<'PY'
import json
with open("data/tooluse_data/train_data.json", "r", encoding="utf-8") as handle:
    print(len(json.load(handle)))
PY
      ;;
    *)
      echo "Unsupported TASK=${TASK}. Expected: tooluse|copa|cb|wsc" >&2
      exit 1
      ;;
  esac
}

TRAIN_SAMPLES="$(resolve_full_train_samples)"
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
RUN_NAME="${TASK}_sft_full_ng${NUM_GENERATIONS}"
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
  --final_eval
  --paper_hparams
  --run_name "${RUN_NAME}"
  --log_input_examples
  --log_examples_eval_only
)

echo "=== SFT FULL ==="
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
