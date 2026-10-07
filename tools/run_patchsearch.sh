#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/pretrained_models/torch}"

# Pass --config and optional overrides through to the Python entry point.
exec "${PYTHON_BIN:?Set PYTHON_BIN to the absolute path of the environment Python}" \
    "${SCRIPT_DIR}/run_patchsearch.py" "$@"
