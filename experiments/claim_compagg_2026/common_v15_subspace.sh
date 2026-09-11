#!/bin/bash
# Shared settings for v15 fixed-subspace routing experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

TRAINER="pretraining_progress/train_clean_mlm_rae_v7_compagg_v14_subspace_route.py"
if [[ ! -f "$TRAINER" ]]; then
  echo "Missing trainer: $TRAINER" >&2
  exit 3
fi

SEED="${SEED:-${1:-0}}"

SYN_PROGRESS="${SYN_PROGRESS:-0.0920386562}"
SYNTAX_PARA_PROGRESS="${SYNTAX_PARA_PROGRESS:-0.5522310170}"
CONTENT_PARA_PROGRESS="${CONTENT_PARA_PROGRESS:-0.6902899227}"
LATE_SYNTAX_PARA_PROGRESS="${LATE_SYNTAX_PARA_PROGRESS:-0.6000000000}"
MID_SYNTAX_PARA_PROGRESS="${MID_SYNTAX_PARA_PROGRESS:-0.2500000000}"
EARLY_SYNTAX_PARA_PROGRESS="${EARLY_SYNTAX_PARA_PROGRESS:-0.1840773125}"

BASE_ARGS=(
  --train_data data/bb26_strictsmall.train
  --valid_data data/bb26_strictsmall.dev
  --tokenizer tokenizers/bb24.model
  --hidden_size 384
  --intermediate_size 1280
  --dropout 0.1
  --batch_size 256
  --lr 0.007
  --epochs 10
  --max_seq_len 0:64,5:256
  --logging_steps 200
  --mlm_prob 0.4
  --mask_decay 0.25
  --mask_replace_prob 0.8
  --random_replace_prob 0.1
  --weight_decay 0.01
  --seed "$SEED"
  --lamb
  --regular_mlm
  --all_checkpoints
)

run_v15_subspace_train() {
  local name="$1"
  local gpu="$2"
  local mechanism="$3"
  shift 3
  local extra_args=("$@")

  mkdir -p log "models/$name"

  echo "============================================================"
  echo "Model: $name"
  echo "Mechanism: $mechanism"
  echo "GPU: ${gpu:-current environment}"
  echo "Seed: $SEED"
  echo "Trainer: $TRAINER"
  echo "Syn progress: $SYN_PROGRESS"
  echo "Early Syntax Para progress: $EARLY_SYNTAX_PARA_PROGRESS"
  echo "Mid Syntax Para progress: $MID_SYNTAX_PARA_PROGRESS"
  echo "Late Syntax Para progress: $LATE_SYNTAX_PARA_PROGRESS"
  echo "Content Para progress: $CONTENT_PARA_PROGRESS"
  echo "============================================================"

  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python "$TRAINER" \
      --output_path "models/$name" \
      "${BASE_ARGS[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee "log/$name.log"
  else
    python "$TRAINER" \
      --output_path "models/$name" \
      "${BASE_ARGS[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee "log/$name.log"
  fi

  cp "$TRAINER" "models/$name/train_code.py"
  cp "$0" "models/$name/run_cmd.sh"

  {
    echo "model=$name"
    echo "seed=$SEED"
    echo "mechanism=$mechanism"
    echo "trainer=$TRAINER"
    echo "recipe=claim_compagg_v15_fixed_subspace_ewok_recovery_2026"
    echo "syn_progress=$SYN_PROGRESS"
    echo "early_syntax_para_progress=$EARLY_SYNTAX_PARA_PROGRESS"
    echo "mid_syntax_para_progress=$MID_SYNTAX_PARA_PROGRESS"
    echo "late_syntax_para_progress=$LATE_SYNTAX_PARA_PROGRESS"
    echo "content_para_progress=$CONTENT_PARA_PROGRESS"
    echo "checkpoint=chck_100M"
    echo "finished_at=$(date)"
  } > "models/$name/claim_manifest.txt"

  test -d "models/$name/chck_100M"
  echo "Done: models/$name"
}
