#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-sdft_tooluse_sweep}"
shift || true

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/sweeps"
LOG_FILE="${LOG_DIR}/tooluse_sweep_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

CMD="cd ${REPO_ROOT} && uv run python scripts/sweep_tooluse.py $* 2>&1 | tee ${LOG_FILE}"

screen -S "${SESSION_NAME}" -dm bash -lc "${CMD}"

echo "Started screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
