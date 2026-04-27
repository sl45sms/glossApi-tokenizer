#!/usr/bin/env bash
# Wrapper: run parquet2json.py inside the uenv virtual environment.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_DIR}/run_uenv.sh" python tools/parquet2json.py "$@"
