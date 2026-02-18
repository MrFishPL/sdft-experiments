#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-wsc}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
DEVICE_BS="${DEVICE_BS:-16}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-8}"
EVAL_STEPS="${EVAL_STEPS:-10}"
EVAL_NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"
REF_MODEL_MIXUP_ALPHA="${REF_MODEL_MIXUP_ALPHA:-0.02}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/runs}"
DISTIL_NUM_GENERATIONS_LIST=(${DISTIL_NUM_GENERATIONS_LIST:-1 4 8 16 32})

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [[ "${TASK}" != "wsc" ]]; then
  echo "This script is designed for TASK=wsc. Override only if intentional. Current TASK=${TASK}" >&2
fi

if (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be >= 1" >&2
  exit 1
fi

EFFECTIVE_BS=$((DEVICE_BS * ACCUM_STEPS * NUM_GPUS))
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

LAUNCH_PREFIX=(uv run accelerate launch --num_processes "${NUM_GPUS}")
if (( NUM_GPUS > 1 )); then
  LAUNCH_PREFIX+=(--multi_gpu)
fi
LAUNCH_PREFIX+=(-m scripts.train)

resolve_train_samples() {
  if [[ -n "${TRAIN_SAMPLES:-}" ]]; then
    echo "${TRAIN_SAMPLES}"
    return
  fi
  case "${TASK}" in
    wsc) echo 554 ;;
    copa) echo 400 ;;
    cb) echo 250 ;;
    *)
      echo "Unknown task '${TASK}'. Set TRAIN_SAMPLES explicitly." >&2
      exit 1
      ;;
  esac
}

TRAIN_SAMPLES="$(resolve_train_samples)"
LONGEST_DISTIL_NUM_GENERATIONS="${DISTIL_NUM_GENERATIONS_LIST[0]}"
for g in "${DISTIL_NUM_GENERATIONS_LIST[@]}"; do
  if (( g > LONGEST_DISTIL_NUM_GENERATIONS )); then
    LONGEST_DISTIL_NUM_GENERATIONS="${g}"
  fi
done

if (( EFFECTIVE_BS <= 0 )); then
  echo "Invalid effective batch size: ${EFFECTIVE_BS}" >&2
  exit 1
fi

# Align SFT baseline step budget to the longest SDFT scenario when MAX_STEPS is not forced.
LONGEST_SDFT_STEPS_PER_EPOCH=$(( (TRAIN_SAMPLES * LONGEST_DISTIL_NUM_GENERATIONS) / EFFECTIVE_BS ))
if (( LONGEST_SDFT_STEPS_PER_EPOCH < 1 )); then
  echo "Computed longest SDFT steps/epoch < 1. Check batch settings." >&2
  exit 1
fi
ALIGNED_SFT_MAX_STEPS=$(( LONGEST_SDFT_STEPS_PER_EPOCH * NUM_EPOCHS ))

if (( MAX_STEPS > 0 )); then
  SFT_MAX_STEPS="${MAX_STEPS}"
  SDFT_MAX_STEPS="${MAX_STEPS}"
else
  SFT_MAX_STEPS="${ALIGNED_SFT_MAX_STEPS}"
  SDFT_MAX_STEPS="-1"
fi

run_scenario() {
  local method="$1"
  local num_generations="$2"
  local scenario_name="$3"
  local scenario_max_steps="$4"

  local run_dir="${OUTPUT_ROOT}/${TASK}_${scenario_name}_${TIMESTAMP}"
  local run_name="${TASK}_${scenario_name}_bs${DEVICE_BS}_acc${ACCUM_STEPS}_eff${EFFECTIVE_BS}"
  local log_file="${LOG_ROOT}/${run_name}.log"

  echo ""
  echo "=== Scenario: ${scenario_name} ==="
  echo "method=${method} num_generations=${num_generations}"
  echo "max_steps=${scenario_max_steps}"
  echo "effective_batch_size=${EFFECTIVE_BS} (= ${DEVICE_BS} * ${ACCUM_STEPS} * ${NUM_GPUS})"
  echo "run_dir=${run_dir}"
  echo "log_file=${log_file}"

  CMD=(
    "${LAUNCH_PREFIX[@]}"
    --method "${method}"
    --task "${TASK}"
    --output_dir "${run_dir}"
    --model_name "${MODEL_NAME}"
    --seed "${SEED}"
    --learning_rate "${LEARNING_RATE}"
    --num_train_epochs "${NUM_EPOCHS}"
    --max_steps "${scenario_max_steps}"
    --per_device_train_batch_size "${DEVICE_BS}"
    --gradient_accumulation_steps "${ACCUM_STEPS}"
    --ref_model_mixup_alpha "${REF_MODEL_MIXUP_ALPHA}"
    --eval_steps "${EVAL_STEPS}"
    --eval_strategy steps
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}"
    --save_steps 100
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --num_generations "${num_generations}"
    --eval_num_generations "${EVAL_NUM_GENERATIONS}"
    --eval_deterministic
    --eval_before_train
    --final_eval
    --log_input_examples
    --log_examples_eval_only
    --warmup_steps "${WARMUP_STEPS}"
    --run_name "${run_name}"
    --paper_hparams
  )

  HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONUNBUFFERED=1 \
    "${CMD[@]}" 2>&1 | tee "${log_file}"
}

# a) classic fine-tuning baseline
echo "Step alignment: train_samples=${TRAIN_SAMPLES} longest_sdft_num_generations=${LONGEST_DISTIL_NUM_GENERATIONS} aligned_sft_max_steps=${ALIGNED_SFT_MAX_STEPS}"
run_scenario "sft" 1 "sft_ep${NUM_EPOCHS}" "${SFT_MAX_STEPS}"

# b-f) sdft with requested num_generations values
for num_generations in "${DISTIL_NUM_GENERATIONS_LIST[@]}"; do
  run_scenario "sdft" "${num_generations}" "sdft_ng${num_generations}_ep${NUM_EPOCHS}" "${SDFT_MAX_STEPS}"
done

echo ""
echo "All WSC scenarios completed successfully."
