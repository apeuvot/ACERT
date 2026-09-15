"""k-means speaker diarization of IEMOCAP conversations.

Produces the json consumed by ``--speaker_diarization_file`` for the
"Speaker-dependent (k-means diarization)" row of the paper: one speaker embedding
per utterance, k-means clustering inside each conversation, and the resulting
speaker groups. The Adjusted Rand Index against the ground-truth speaker of each
utterance (the _F / _M tag of the IEMOCAP filenames) is reported as a sanity check.

Output files, written in ``--output_path``:
  * ``speaker_diarization_preds.json`` : {conversation: [[utt, ...], [utt, ...]]}
  * ``results.json``                   : per-conversation and mean ARI
"""

import os
import json
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def get_true_labels(conversations):
    """Ground-truth speaker of each utterance, from the _F / _M tag of its name."""
    true_labels = {}
    for conv, files in conversations.items():
        labels = []
        for f in files:
            # "XX" means no label (and the audio file is not available)
            if "_F" in f and 'XX' not in f:
                labels.append(1)  # 1 for F
            elif "_M" in f and 'XX' not in f:
                labels.append(0)  # 0 for M
        true_labels[conv] = labels
    return true_labels


def load_embedder(model_name):
    """Return a callable wav_path -> embedding."""
    if 'pyannote' in model_name:
        from pyannote.audio import Inference, Model
        model = Model.from_pretrained(model_name)
        inference = Inference(model, window="whole")
        return inference

    import wespeaker
    # Needs a local folder. To download it from the Hub:
    #   from huggingface_hub import snapshot_download
    #   snapshot_download("Wespeaker/wespeaker-voxceleb-resnet34-LM")
    model = wespeaker.load_model(model_name)
    return model.extract_embedding


def main(args):
    with open(args.conversations_file, "r") as f:
        conversations = json.load(f)

    embed = load_embedder(args.model_name)
    true_labels = get_true_labels(conversations)

    ari_scores = {}
    diarization_preds = {}

    for conv, files in conversations.items():
        embeddings = []
        utterances_files = []
        for f in files:
            # 'XX' means no label in IEMOCAP => the wav file does not exist
            if 'XX' in f:
                continue
            embeddings.append(embed(os.path.join(args.data_dir, f + ".wav")))
            utterances_files.append(f)

        embeddings = np.vstack(embeddings)

        clustering = KMeans(n_clusters=args.n_clusters, random_state=0, n_init="auto")
        pred_labels = clustering.fit_predict(embeddings)

        ari = adjusted_rand_score(true_labels[conv], pred_labels)
        ari_scores[conv] = ari
        print(f"Conversation {conv}: ARI = {ari:.4f}")

        speaker_groups = {}
        for file_name, spk in zip(utterances_files, pred_labels):
            speaker_groups.setdefault(int(spk), []).append(file_name)

        diarization_preds[conv] = [speaker_groups[k] for k in sorted(speaker_groups)]

    mean_ari = float(np.mean(list(ari_scores.values())))
    print(f"\nMean ARI over all conversations: {mean_ari:.4f}")

    os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.output_path, "results.json"), "w") as f:
        json.dump({"ari_scores": ari_scores, "mean_ari": mean_ari}, f, indent=2)

    diarization_path = os.path.join(args.output_path, "speaker_diarization_preds.json")
    with open(diarization_path, "w") as f:
        json.dump(diarization_preds, f, indent=2)

    print(f"Results saved to {args.output_path}")
    print(f"Pass {diarization_path} to --speaker_diarization_file to train on it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, required=True,
                        help="speaker embedding model: a local wespeaker folder (the paper "
                             "uses Wespeaker/wespeaker-voxceleb-resnet34-LM) or a pyannote id")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="folder containing the 16 kHz wav files")
    parser.add_argument("--conversations_file", type=str, required=True,
                        help="Dataset/IEMOCAP/conversations_orders.json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="folder where the predictions and ARI scores are written")
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="number of speakers per conversation (2 for the dyadic IEMOCAP)")
    args = parser.parse_args()

    main(args)
