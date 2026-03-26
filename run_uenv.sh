#!/usr/bin/env bash

set -euo pipefail

IMAGE="prgenv-gnu/24.11:v1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${REPO_DIR}/.venv-uenv"
ENV_FILE="${REPO_DIR}/.env"

if ! command -v uenv >/dev/null 2>&1; then
  echo "uenv is not available in PATH." >&2
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  cat >&2 <<EOF
Missing virtual environment at ${VENV_PATH}.

Create it with:
uenv image pull ${IMAGE}
uenv run ${IMAGE} --view=default -- bash -lc '
cd ${REPO_DIR}
python3 -m venv .venv-uenv
source .venv-uenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
'
EOF
  exit 1
fi

if [[ $# -eq 0 ]]; then
  cat >&2 <<EOF
Usage: ./run_uenv.sh <command> [args...]

Examples:
./run_uenv.sh python scripts/extract_apertus_tokenizer.py --trust-remote-code
./run_uenv.sh python scripts/compare_tokenizers.py --trust-remote-code --sample-file greek_samples.txt
./run_uenv.sh bash
EOF
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

escaped_args=()
for arg in "$@"; do
  escaped_args+=("$(printf '%q' "$arg")")
done

command_string="cd ${REPO_DIR} && source .venv-uenv/bin/activate && ${escaped_args[*]}"

exec uenv run "${IMAGE}" --view=default -- bash -lc "${command_string}"