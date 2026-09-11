#!/bin/bash
# v22 B: stronger syntax-Para, but reduce Syn on syntax tokens after handoff.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_v15_subspace.sh"

GPU="${COMPAGG_V22_V15_RELAY_SYNP050_SYNKEEP010_GPU:-${GPU:-}}"
NAME="bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s${SEED}"

run_v15_subspace_train "$NAME" "$GPU" "v22_v15_relay_synp050_synkeep010" \
  --rae_syn \
  --rae_syn_weight 0.0005 \
  --rae_syn_topk 8 \
  --rae_syn_layer -1 \
  --rae_syn_temp 1.0 \
  --rae_syn_warmup_progress "$SYN_PROGRESS" \
  --rae_syn_exclude_entity \
  --rae_syn_syntax_after_syntax_para_scale 0.10 \
  --rae_para \
  --rae_para_weight 0.000001 \
  --rae_para_topk 8 \
  --rae_para_temp 1.0 \
  --rae_para_warmup_progress 0.9900000000 \
  --rae_para_ramp_progress 0.0 \
  --rae_para_entity_weight 0.0 \
  --rae_para_reading_weight 0.0 \
  --rae_syn_subspace 0:256 \
  --rae_para_subspace 256:384 \
  --rae_para_syntax_weight 0.000050 \
  --rae_para_syntax_warmup_progress "$EARLY_SYNTAX_PARA_PROGRESS" \
  --entity_repr_anchor_weight 0.00025 \
  --entity_repr_anchor_start_progress "$SYN_PROGRESS" \
  --entity_repr_anchor_end_progress "$LATE_SYNTAX_PARA_PROGRESS"
