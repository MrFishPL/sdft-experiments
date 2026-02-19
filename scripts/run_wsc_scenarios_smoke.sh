#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NUM_EPOCHS="${NUM_EPOCHS:-1}" \
MAX_STEPS="${MAX_STEPS:-2}" \
EVAL_STEPS="${EVAL_STEPS:-1}" \
DEVICE_BS="${DEVICE_BS:-2}" \
ACCUM_STEPS="${ACCUM_STEPS:-2}" \
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-4}" \
bash scripts/run_wsc_scenarios.sh
