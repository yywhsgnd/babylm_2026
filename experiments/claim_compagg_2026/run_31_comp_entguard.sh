#!/bin/bash
# v3 composition-only: Syn axis with entity-only protection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_single_axis_v3.sh"

GPU="${COMP_ENTGUARD_GPU:-${GPU:-}}"
NAME="bb26_claim_comp_entguard_s${SEED}"

run_single_axis_v3_train "$NAME" "$GPU" "composition_syn_entity_guard_weight_0005" \
  --rae_syn \
  --rae_syn_weight 0.0005 \
  --rae_syn_topk 8 \
  --rae_syn_layer -1 \
  --rae_syn_temp 1.0 \
  --rae_syn_warmup_progress "$SYN_PROGRESS" \
  --rae_syn_exclude_entity
