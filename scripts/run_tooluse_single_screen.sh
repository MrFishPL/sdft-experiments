#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-sdft_tooluse_single}"
shift || true

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/runs"
RUN_DIR="${REPO_ROOT}/runs/tooluse_paper_single_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/tooluse_single_${TIMESTAMP}.log"
RUN_NAME="tooluse_paper_single_lr1e-5_ep2_bs32_alpha0.02_s42"

mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_DIR}"

CMD="cd ${REPO_ROOT} && PYTHONUNBUFFERED=1 uv run python -u -m scripts.train \
  --output_dir ${RUN_DIR} \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --seed 42 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --ref_model_mixup_alpha 0.02 \
  --eval_steps 100 \
  --eval_strategy steps \
  --per_device_eval_batch_size 8 \
  --save_steps 100 \
  --max_prompt_length 1024 \
  --max_completion_length 2048 \
  --num_generations 1 \
  --warmup_steps 10 \
  --run_name ${RUN_NAME} \
  --paper_hparams \
  $* 2>&1 | tee ${LOG_FILE}"

screen -S "${SESSION_NAME}" -dm bash -lc "${CMD}"

echo "Started screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Log: ${LOG_FILE}"
