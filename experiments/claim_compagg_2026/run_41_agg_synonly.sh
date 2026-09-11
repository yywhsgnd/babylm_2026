#!/bin/bash
# Aggregation-axis v4: keep only syntax Para; content Para is effectively off.
# Purpose: test whether aggregation gains can be kept without damaging Entity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

GPU="${AGG_SYNONLY_GPU:-${GPU:-}}"
NAME="bb26_claim_agg_synonly_s${SEED}"

run_claim_train "$NAME" "$GPU" \
  --rae_para \
  --rae_para_weight 0.000001 \
  --rae_para_topk 8 \
  --rae_para_temp 1.0 \
  --rae_para_warmup_progress 0.9900000000 \
  --rae_para_ramp_progress 0.0 \
  --rae_para_syntax_weight 0.00005 \
  --rae_para_syntax_warmup_progress "$SYNTAX_PARA_PROGRESS" \
  --rae_para_entity_weight 0.0 \
  --rae_para_reading_weight 0.0
