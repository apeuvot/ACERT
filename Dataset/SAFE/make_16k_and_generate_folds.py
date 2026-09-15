import pandas as pd
import json
import random
import argparse
import soundfile as sf
import resampy
from pathlib import Path
from collections import defaultdict

def process_safe_dataset(safe_dir, seed=0):
    """
    1. Segments audio files based on annotations.
    2. Resamples segments to 16kHz mono.
    3. Groups files by 'film' and generates cross-validation folds in JSON format.
    """
    safe_path = Path(safe_dir)
    
    # Configuration and Paths
    annotation_file = safe_path / "AnnotationSegmentCoder3.txt"
    audio_output_path = Path("Audio_16k")
    labels_output_path = Path("labels_sess")
    
    audio_output_path.mkdir(exist_ok=True)
    labels_output_path.mkdir(exist_ok=True)

    if not annotation_file.exists():
        print(f"Error: Annotation file not found at {annotation_file}")
        return

    # Load annotations
    try:
        df = pd.read_csv(annotation_file, sep='\t', encoding='latin-1')
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return

    audio_emotions = {}

    # --- Part 1: Audio Processing ---
    print("Starting audio segmentation and resampling...")
    for seq_folder in safe_path.iterdir():
        if not seq_folder.is_dir() or seq_folder.name == "Audio_16k":
            continue
        
        for sub_folder in seq_folder.iterdir():
            if not sub_folder.is_dir():
                continue
            
            wav_files = list(sub_folder.glob("*.wav"))
            if not wav_files:
                continue
            
            # Process the primary wav file in the sub-directory
            wav_file = wav_files[0]
            wav_name = wav_file.stem

            data, samplerate = sf.read(wav_file)

            # Convert to mono
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Resample to 16kHz
            target_rate = 16000
            if samplerate != target_rate:
                data = resampy.resample(data, samplerate, target_rate)
                samplerate = target_rate

            # Filter segments for this specific file
            mask = df['file'].str.startswith(wav_name)
            df_segments = df[mask]

            for idx, row in enumerate(df_segments.itertuples(), start=1):
                start_sample = int(row.timestart * samplerate)
                end_sample = int(row.timeend * samplerate)

                segment_data = data[start_sample:end_sample]
                new_fname = f"{wav_name}_{idx:04d}.wav"
                
                sf.write(audio_output_path / new_fname, segment_data, samplerate)

                # Collect emotion labels (excluding placeholders)
                if hasattr(row, 'emotion') and row.emotion not in ['none', 'PAS_REMPLI']:
                    audio_emotions[new_fname] = row.emotion

    print(f"Audio processing complete. {len(audio_emotions)} segments created.")

    # --- Part 2: Fold Generation ---
    print("\nGenerating cross-validation folds...")
    
    # Group by film (prefix before the first underscore)
    film_to_files = defaultdict(dict)
    for fname, emotion in audio_emotions.items():
        film = fname.split("_")[0]
        film_to_files[film][fname] = emotion

    films = sorted(film_to_files.keys())
    
    # Validation check for the number of expected films
    if len(films) != 30:
        print(f"Warning: Expected 30 films, but found {len(films)}.")
    
    # Shuffle to avoid ordering bias (seeded so the folds are reproducible)
    random.Random(seed).shuffle(films)

    # Fold configuration
    n_test_films = 6
    n_folds = len(films) // n_test_films

    for fold_idx in range(n_folds):
        test_films = films[fold_idx * n_test_films : (fold_idx + 1) * n_test_films]
        train_films = [f for f in films if f not in test_films]

        train_dict = {}
        test_dict = {}

        # Populate train set
        for film in train_films:
            train_dict.update(film_to_files[film])

        # Populate test set
        for film in test_films:
            test_dict.update(film_to_files[film])

        # Save to JSON
        label_data = {
            "Train": train_dict,
            "Test": test_dict
        }

        out_path = labels_output_path / f"label_{fold_idx + 1}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)

        # Updated print statement to show the list of films in the test set
        print(f"--- Fold {fold_idx + 1} ---")
        print(f"Test films: {', '.join(test_films)}")
        print(f"Stats: {len(train_dict)} train files | {len(test_dict)} test files")
        print(f"Saved to: {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SAFE dataset: segment audio and generate CV folds.")
    parser.add_argument("--safe_dir", type=str, required=True, 
                        help="Path to the root SAFE dataset directory")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the random partition of the 30 movies into 5 folds")

    args = parser.parse_args()
    process_safe_dataset(args.safe_dir, seed=args.seed)