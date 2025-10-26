#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

abort() {
    echo "error: $*" >&2
    exit 1
}

activate_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        if [[ -d "${VENV_DIR}" ]]; then
            # shellcheck disable=SC1091
            source "${VENV_DIR}/bin/activate"
        elif [[ -n "${CI:-}" ]]; then
            echo "Using system interpreter provided by CI"
        else
            abort "virtual environment not found. run scripts/bootstrap first."
        fi
    fi
}

run_python_module() {
    activate_venv
    python -m "$@"
}

install_dev_dependencies() {
    activate_venv
    pip install --upgrade pip
    pip install -e ".[dev]"
}
