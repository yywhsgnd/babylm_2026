#!/bin/bash
# Shared settings for v3 composition-axis claim experiments.
# These create new model names and keep the fixed clean baseline untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

TRAINER="pretraining_progress/train_clean_mlm_rae_v7_compagg_progress.py"
if [[ ! -f "$TRAINER" ]]; then
  echo "Missing trainer: $TRAINER" >&2
  exit 3
fi

SEED="${SEED:-${1:-0}}"

# Progress gates from the 1000/6000/7500-step recipes.
SYN_PROGRESS="${SYN_PROGRESS:-0.0920386562}"              # old step 1000 / 10865
PARA_PROGRESS="${PARA_PROGRESS:-0.5522310170}"            # old step 6000 / 10865
LATE_PARA_PROGRESS="${LATE_PARA_PROGRESS:-0.6902899227}"  # old step 7500 / 10865

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

run_single_axis_v3_train() {
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
  echo "Para progress: $PARA_PROGRESS"
  echo "Late Para progress: $LATE_PARA_PROGRESS"
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
    echo "recipe=claim_single_axis_v3_comp_entity_guard_sweep_2026"
    echo "syn_progress=$SYN_PROGRESS"
    echo "para_progress=$PARA_PROGRESS"
    echo "late_para_progress=$LATE_PARA_PROGRESS"
    echo "checkpoint=chck_100M"
    echo "finished_at=$(date)"
  } > "models/$name/claim_manifest.txt"

  test -d "models/$name/chck_100M"
  echo "Done: models/$name"
}
