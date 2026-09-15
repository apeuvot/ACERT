import json
from pathlib import Path
import re
import argparse

def generate_metalabels(iemocap_dir):
    iemocap_path = Path(iemocap_dir)
    print('Generating metalabels...')
    
    metalabel = {}
    for i in range(5):
        sess = i + 1
        label_dir = iemocap_path / "IEMOCAP_full_release" / f"Session{sess}" / "dialog" / "EmoEvaluation"
        for labelfile in label_dir.rglob('*.txt'):
            with open(labelfile, 'r') as f:
                for line in f.readlines():
                    # Extract file name and emotion label
                    m = re.match(r".*(Ses.*)\t(.*)\t.*", line)
                    if m:
                        name, label = m.groups()
                        metalabel[name + '.wav'] = label
                        
    # Save the consolidated labels to a JSON file
    with open('metalabel.json', 'w') as f:
        json.dump(metalabel, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metalabels for IEMOCAP")
    parser.add_argument("--iemocap_dir", type=str, required=True, help="Path to IEMOCAP directory")
    args = parser.parse_args()

    generate_metalabels(args.iemocap_dir)