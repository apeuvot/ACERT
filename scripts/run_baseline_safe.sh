#!/bin/bash
# Utterance-level baseline on SAFE: HuBERT-large fine-tuning, no conversational context.
set -e

MODEL_PATH="${MODEL_PATH:-facebook/hubert-large-ll60k}"
MODEL_SHORT_NAME="${MODEL_PATH##*/}"

DATASET_NAME="SAFE"
BASE_DATA_DIR="./Dataset/${DATASET_NAME}"
DATA_DIR="${BASE_DATA_DIR}/Audio_16k"
LABEL_DIR="${BASE_DATA_DIR}/labels_sess"

LR=1.3e-4
BATCH_SIZE=16
MAX_EPOCHS=50
STRATEGY="${STRATEGY:-DDP}"   # only used when several GPUs are visible

DATETIME=$(date +"%d-%m-%Y_%H-%M")
SAVING_PATH="./results/${DATASET_NAME}/${MODEL_SHORT_NAME}_baseline__${DATETIME}"

python run_downstream_custom_multiple_fold.py \
    --precision 16 \
    --datadir "$DATA_DIR" \
    --labeldir "$LABEL_DIR" \
    --dataset_name "$DATASET_NAME" \
    --model_path "$MODEL_PATH" \
    --saving_path "$SAVING_PATH" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --training_strategy "$STRATEGY" \
    --num_exps 5
