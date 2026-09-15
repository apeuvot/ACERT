#!/bin/bash
# ACERT on MELD, t = 10 s. Set CONTEXT_DURATION to sweep the context window.
set -e

MODEL_PATH="${MODEL_PATH:-facebook/hubert-large-ll60k}"
MODEL_SHORT_NAME="${MODEL_PATH##*/}"

DATASET_NAME="MELD"
BASE_DATA_DIR="./Dataset/${DATASET_NAME}"
DATA_DIR="${BASE_DATA_DIR}/Audio_16k"
LABEL_DIR="${BASE_DATA_DIR}/labels_sess"

LR=5e-5
BATCH_SIZE=16
MAX_EPOCHS=50
STRATEGY="${STRATEGY:-DDP}"   # only used when several GPUs are visible

CONTEXT_DURATION="${CONTEXT_DURATION:-10}"
CONTEXT_DIRECTION="past"
CONTEXT_SOURCE="recent"

DATETIME=$(date +"%d-%m-%Y_%H-%M")
SAVING_PATH="./results/${DATASET_NAME}/${MODEL_SHORT_NAME}_t=${CONTEXT_DURATION}_acert__${DATETIME}"

python run_downstream_custom_multiple_fold.py \
    --precision 16 \
    --datadir "$DATA_DIR" \
    --labeldir "$LABEL_DIR" \
    --dataset_name "$DATASET_NAME" \
    --model_path "$MODEL_PATH" \
    --saving_path "$SAVING_PATH" \
    --context_duration "$CONTEXT_DURATION" \
    --context_direction "$CONTEXT_DIRECTION" \
    --context_source "$CONTEXT_SOURCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --training_strategy "$STRATEGY" \
    --num_exps 5
