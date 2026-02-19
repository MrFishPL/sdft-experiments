#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-2}" \
EVAL_STEPS="${EVAL_STEPS:-1}" \
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-4}" \
LOG_INPUT_EXAMPLES="${LOG_INPUT_EXAMPLES:-0}" \
DISTIL_NUM_GENERATIONS_LIST="${DISTIL_NUM_GENERATIONS_LIST:-256 128 64 32 16 8 4 1}" \
bash scripts/run_superglue_fewshot_scenarios.sh
