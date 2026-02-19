#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASKS=(${TASKS:-copa cb wsc})
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
DEVICE_BS="${DEVICE_BS:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-8}"
EVAL_STEPS="${EVAL_STEPS:-10}"
LOG_INPUT_EXAMPLES="${LOG_INPUT_EXAMPLES:-1}"
LOG_EXAMPLES_EVAL_ONLY="${LOG_EXAMPLES_EVAL_ONLY:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"
REF_MODEL_MIXUP_ALPHA="${REF_MODEL_MIXUP_ALPHA:-0.02}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/runs}"
FEWSHOT_INDICES_FILE="${FEWSHOT_INDICES_FILE:-${REPO_ROOT}/data/superglue_fewshot_5shot_curated.json}"
FEWSHOT_NUM_EXAMPLES="${FEWSHOT_NUM_EXAMPLES:-5}"
DISTIL_NUM_GENERATIONS_LIST=(${DISTIL_NUM_GENERATIONS_LIST:-256 128 64 32 16 8 4 1})
EPOCHS_AT_MAX_GEN="${EPOCHS_AT_MAX_GEN:-5}"
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:--1}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

if (( NUM_GPUS != 1 || DEVICE_BS != 1 || ACCUM_STEPS != 1 )); then
  echo "This script requires NUM_GPUS=1, DEVICE_BS=1, ACCUM_STEPS=1 for exact step comparability." >&2
  exit 1
fi
if (( FEWSHOT_NUM_EXAMPLES < 1 )); then
  echo "FEWSHOT_NUM_EXAMPLES must be >= 1" >&2
  exit 1
fi
if (( EPOCHS_AT_MAX_GEN < 1 )); then
  echo "EPOCHS_AT_MAX_GEN must be >= 1" >&2
  exit 1
fi
if [[ ! -f "${FEWSHOT_INDICES_FILE}" ]]; then
  echo "Few-shot index file not found: ${FEWSHOT_INDICES_FILE}" >&2
  exit 1
fi

G_MAX="${DISTIL_NUM_GENERATIONS_LIST[0]}"
if (( G_MAX < 1 )); then
  echo "Invalid first generation value in DISTIL_NUM_GENERATIONS_LIST: ${G_MAX}" >&2
  exit 1
fi

REFERENCE_UPDATES=$(( FEWSHOT_NUM_EXAMPLES * G_MAX * EPOCHS_AT_MAX_GEN ))
BASELINE_MAX_STEPS="${REFERENCE_UPDATES}"
if (( MAX_STEPS_OVERRIDE > 0 )); then
  BASELINE_MAX_STEPS="${MAX_STEPS_OVERRIDE}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

LAUNCH_PREFIX=(uv run accelerate launch --num_processes "${NUM_GPUS}")
LAUNCH_PREFIX+=(-m scripts.train)

run_scenario() {
  local task="$1"
  local method="$2"
  local num_generations="$3"
  local scenario_epochs="$4"
  local scenario_max_steps="$5"
  local extra_name="$6"

  local run_dir="${OUTPUT_ROOT}/${task}_${extra_name}_${TIMESTAMP}"
  local run_name="${task}_${extra_name}"
  local log_file="${LOG_ROOT}/${run_name}_${TIMESTAMP}.log"

  echo ""
  echo "=== Scenario: ${extra_name} ==="
  echo "task=${task} method=${method} num_generations=${num_generations}"
  echo "epochs=${scenario_epochs} max_steps=${scenario_max_steps} eval_steps=${EVAL_STEPS}"
  echo "run_dir=${run_dir}"
  echo "log_file=${log_file}"

  CMD=(
    "${LAUNCH_PREFIX[@]}"
    --method "${method}"
    --task "${task}"
    --output_dir "${run_dir}"
    --model_name "${MODEL_NAME}"
    --seed "${SEED}"
    --learning_rate "${LEARNING_RATE}"
    --num_train_epochs "${scenario_epochs}"
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
    --eval_deterministic
    --eval_before_train
    --final_eval
    --warmup_steps "${WARMUP_STEPS}"
    --run_name "${run_name}"
    --paper_hparams
    --fewshot_indices_file "${FEWSHOT_INDICES_FILE}"
    --fewshot_num_examples "${FEWSHOT_NUM_EXAMPLES}"
  )

  if [[ "${LOG_INPUT_EXAMPLES}" == "1" ]]; then
    CMD+=(--log_input_examples)
  else
    CMD+=(--no-log_input_examples)
  fi
  if [[ "${LOG_EXAMPLES_EVAL_ONLY}" == "1" ]]; then
    CMD+=(--log_examples_eval_only)
  else
    CMD+=(--no-log_examples_eval_only)
  fi

  if [[ "${method}" == "sdft" ]]; then
    CMD+=(--distil_generation_batch_size "${num_generations}")
  fi

  HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONUNBUFFERED=1 \
    "${CMD[@]}" 2>&1 | tee "${log_file}"
}

echo "Few-shot schedule summary"
echo "tasks=${TASKS[*]}"
echo "fewshot_num_examples=${FEWSHOT_NUM_EXAMPLES}"
echo "g_max=${G_MAX} epochs_at_g_max=${EPOCHS_AT_MAX_GEN}"
echo "reference_updates_per_run=${REFERENCE_UPDATES}"
echo "baseline_max_steps=${BASELINE_MAX_STEPS}"
if (( MAX_STEPS_OVERRIDE > 0 )); then
  echo "MAX_STEPS_OVERRIDE is active -> all scenarios use max_steps=${MAX_STEPS_OVERRIDE}"
fi

for task in "${TASKS[@]}"; do
  echo ""
  echo "##### Task: ${task} #####"

  run_scenario "${task}" "sft" 1 "${EPOCHS_AT_MAX_GEN}" "${BASELINE_MAX_STEPS}" "${task}_fewshot5_sft_baseline"

  for g in "${DISTIL_NUM_GENERATIONS_LIST[@]}"; do
    if (( g < 1 )); then
      echo "Invalid num_generations value: ${g}" >&2
      exit 1
    fi
    if (( (EPOCHS_AT_MAX_GEN * G_MAX) % g != 0 )); then
      echo "EPOCHS_AT_MAX_GEN * G_MAX must be divisible by g. Got ${EPOCHS_AT_MAX_GEN} * ${G_MAX} / ${g}" >&2
      exit 1
    fi
    scenario_epochs=$(( (EPOCHS_AT_MAX_GEN * G_MAX) / g ))
    scenario_max_steps="-1"
    if (( MAX_STEPS_OVERRIDE > 0 )); then
      scenario_max_steps="${MAX_STEPS_OVERRIDE}"
    fi
    expected_updates=$(( FEWSHOT_NUM_EXAMPLES * g * scenario_epochs ))
    echo "Planned sdft g=${g}: epochs=${scenario_epochs}, expected_updates=${expected_updates}"

    run_scenario \
      "${task}" \
      "sdft" \
      "${g}" \
      "${scenario_epochs}" \
      "${scenario_max_steps}" \
      "${task}_fewshot5_sdft_ng${g}"
  done
done

echo ""
echo "All SuperGLUE few-shot scenarios completed successfully."
