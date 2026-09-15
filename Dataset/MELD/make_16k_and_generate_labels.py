import os
import pandas as pd
import json
import librosa
import soundfile as sf
import argparse
from pathlib import Path

def load_annotations(csv_file, split_name, video_dir):
    """
    Loads a MELD CSV file and returns a dictionary {filename: emotion}.
    Verifies if the corresponding mp4 file exists.
    """
    df = pd.read_csv(csv_file)
    data = {}
    
    for _, row in df.iterrows():
        dialogue_id = int(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])
        emotion = row["Emotion"].strip().lower()
        
        # Target filename
        filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        filepath = os.path.join(video_dir, filename)
        
        if os.path.exists(filepath):
            data[filename] = emotion
        else:
            print(f"⚠️ Missing file ({split_name}): {filename}")
    
    return data

def convert_mp4_to_wav_16k(mp4_path, wav_path, corrupted_list):
    """
    Converts MP4 to WAV using librosa (auto-resample to 16kHz and mono).
    """
    try:
        audio, sr = librosa.load(mp4_path, sr=16000, mono=True)
        sf.write(wav_path, audio, 16000)
    except Exception as e:
        print(f"⚠️ Failed to convert {mp4_path}: {e}")
        corrupted_list.append(str(mp4_path))

def process_audio_files(data_dict, split, video_dir, audio_dir, corrupted_list):
    """
    Processes all files in a dictionary, converting them and updating the keys.
    """
    processed_data = {}
    for filename, emotion in data_dict.items():
        mp4_path = os.path.join(video_dir, filename)
        
        # New filename format: {split}_diaX_uttY.wav
        wav_filename = f"{split.lower()}_{filename.replace('.mp4', '.wav')}"
        wav_path = os.path.join(audio_dir, wav_filename)
        
        if not os.path.exists(wav_path):
            convert_mp4_to_wav_16k(mp4_path, wav_path, corrupted_list)
        
        processed_data[wav_filename] = emotion
    return processed_data

def main(meld_dir):
    # Setup Paths
    dataset_path = Path(meld_dir) / "MELD.Raw"
    audio_dir = Path("./Audio_16k")
    label_dir = Path("./labels_sess")
    
    audio_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    # Define splits configuration
    splits_config = {
        "Train": {
            "video": dataset_path / "train_splits",
            "csv": dataset_path / "train_sent_emo.csv"
        },
        "Val": {
            "video": dataset_path / "dev_splits_complete",
            "csv": dataset_path / "dev_sent_emo.csv"
        },
        "Test": {
            "video": dataset_path / "output_repeated_splits_test",
            "csv": dataset_path / "test_sent_emo.csv"
        }
    }

    corrupted_files = []
    final_dataset_dict = {}

    for split_name, paths in splits_config.items():
        print(f"Processing {split_name} split...")
        
        # 1. Load annotations
        raw_data = load_annotations(paths["csv"], split_name, paths["video"])
        
        # 2. Handle known corrupted file in Train split
        target_corrupted = "dia125_utt3.mp4"
        if split_name == "Train" and target_corrupted in raw_data:
            print(f"ℹ️ Removing known corrupted file: {target_corrupted}")
            corrupted_files.append(str(paths["video"] / target_corrupted))
            del raw_data[target_corrupted]

        # 3. Convert and rename
        processed_split_data = process_audio_files(
            raw_data, split_name, paths["video"], audio_dir, corrupted_files
        )
        
        final_dataset_dict[split_name] = processed_split_data

    # Save JSON labels
    json_output_path = label_dir / "label.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Conversion complete. Audio saved in {audio_dir}")
    print(f"✅ Labels saved in {json_output_path}")
    
    if corrupted_files:
        print(f"\nSummary of excluded/corrupted files ({len(corrupted_files)}):")
        for f in corrupted_files:
            print(f" - {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MELD dataset: Convert MP4 to 16kHz WAV and generate labels.")
    parser.add_argument("--meld_dir", type=str, required=True, help="Path to the MELD root directory")
    
    args = parser.parse_args()
    main(args.meld_dir)