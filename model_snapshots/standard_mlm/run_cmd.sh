#!/bin/bash
# Baseline: clean MLM only, using the same progress-capable trainer as all claim models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

GPU="${CLEAN_GPU:-${GPU:-}}"
NAME="bb26_claim_clean_s${SEED}"

run_claim_train "$NAME" "$GPU"
