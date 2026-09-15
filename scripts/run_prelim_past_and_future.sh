#!/bin/bash
# Preliminary design experiment: context taken on both sides of the target.
set -e

MODEL_PATH="${MODEL_PATH:-facebook/hubert-large-ll60k}"
MODEL_SHORT_NAME="${MODEL_PATH##*/}"

DATASET_NAME="IEMOCAP"
BASE_DATA_DIR="./Dataset/${DATASET_NAME}"
DATA_DIR="${BASE_DATA_DIR}/Audio_16k"
LABEL_DIR="${BASE_DATA_DIR}/labels_sess"
CONV_ORDER_FILE="${BASE_DATA_DIR}/conversations_orders.json"

# same setting as the main IEMOCAP run, for comparability
CONTEXT_DURATION="${CONTEXT_DURATION:-25}"
LR=1.3e-4
BATCH_SIZE=16
MAX_EPOCHS=15
STRATEGY="${STRATEGY:-DDP}"

DATETIME=$(date +"%d-%m-%Y_%H-%M")
SAVING_PATH="./results/${DATASET_NAME}/${MODEL_SHORT_NAME}_t=${CONTEXT_DURATION}_past_and_future__${DATETIME}"

python run_downstream_custom_multiple_fold.py \
    --precision 16 \
    --datadir "$DATA_DIR" \
    --labeldir "$LABEL_DIR" \
    --dataset_name "$DATASET_NAME" \
    --model_path "$MODEL_PATH" \
    --saving_path "$SAVING_PATH" \
    --context_duration "$CONTEXT_DURATION" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --training_strategy "$STRATEGY" \
    --num_exps 5 \
    --context_direction past_and_future \
    --context_source recent \
    --conv_order_file "$CONV_ORDER_FILE"
