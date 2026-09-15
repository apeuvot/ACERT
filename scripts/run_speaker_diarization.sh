#!/bin/bash
# k-means speaker diarization of the IEMOCAP conversations, used by
# scripts/run_prelim_speaker_dependent_diarization.sh.
#
# The paper uses the WeSpeaker ResNet34 embeddings. Download them once with
#   python -c "from huggingface_hub import snapshot_download; \
#              print(snapshot_download('Wespeaker/wespeaker-voxceleb-resnet34-LM'))"
# and point EMBEDDING_MODEL at the printed folder.
set -e

EMBEDDING_MODEL="${EMBEDDING_MODEL:?set EMBEDDING_MODEL to a local wespeaker folder or a pyannote model id}"

DATASET_NAME="IEMOCAP"
BASE_DATA_DIR="./Dataset/${DATASET_NAME}"

python speaker_diarization.py \
    --model_name "$EMBEDDING_MODEL" \
    --data_dir "${BASE_DATA_DIR}/Audio_16k" \
    --conversations_file "${BASE_DATA_DIR}/conversations_orders.json" \
    --output_path "./results/${DATASET_NAME}/speaker_diarization" \
    --n_clusters 2
