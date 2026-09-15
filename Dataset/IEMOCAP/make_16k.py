import librosa
import os
import soundfile as sf
from pathlib import Path
import argparse

def process_iemocap(iemocap_dir):
    Path('Audio_16k').mkdir(exist_ok=True)
    root_dir = Path(iemocap_dir)
    
    print("Downsampling IEMOCAP to 16k")
    
    for i in range(5):
        sess = i + 1
        current_dir = root_dir / "IEMOCAP_full_release" / f"Session{sess}" / "sentences" / "wav"
        
        for full_audio_name in current_dir.rglob('*.wav'):
            audio, sr = librosa.load(str(full_audio_name), sr=None)
            audio_name = full_audio_name.name
            
            # Verify sample rate and save
            assert sr == 16000
            sf.write(os.path.join('Audio_16k', audio_name), audio, 16000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEMOCAP preprocessing")
    parser.add_argument("--iemocap_dir", type=str, required=True, help="Path to IEMOCAP root")
    args = parser.parse_args()

    process_iemocap(args.iemocap_dir)