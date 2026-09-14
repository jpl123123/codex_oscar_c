#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$PROJECT_ROOT"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export OSCAR_ASCEND_ENABLED=0
exec "${OSCAR_PYTHON:-python3}" -m oscar_ascend.deploy "$@"
