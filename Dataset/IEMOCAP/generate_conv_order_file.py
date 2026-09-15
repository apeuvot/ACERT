import os
import re
import argparse
import json

def extract_conversations(iemocap_dir, output_json="conversations_orders.json"):
    """
    Traverses the root_dir to find IEMOCAP transcriptions.
    Extracts utterances in file order and writes a JSON:
        { conversation_name : [utterance_id_1, utterance_id_2, ...] }
    """
    # regex to recognize an utterance line
    utt_pattern = re.compile(r"^(Ses\d+[FM]_.+?)\s*\[")

    conversations = {}

    root_dir = os.path.join(iemocap_dir, "IEMOCAP_full_release")

    for session in os.listdir(root_dir):
        session_path = os.path.join(root_dir, session)

        # Keep only SessionX folders
        if not os.path.isdir(session_path) or not session.lower().startswith("session"):
            continue

        trans_dir = os.path.join(session_path, "dialog", "transcriptions")
        if not os.path.isdir(trans_dir):
            continue

        # Read .txt files
        for fname in os.listdir(trans_dir):
            if not fname.endswith(".txt"):
                continue

            conv_name = fname.replace(".txt", "")
            conv_path = os.path.join(trans_dir, fname)

            utterances = []

            with open(conv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    m = utt_pattern.match(line)
                    if m:
                        utterances.append(m.group(1))

            conversations[conv_name] = utterances

    # Write JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete. Results saved in {output_json}")
    return conversations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract conversation utterance orders from IEMOCAP transcriptions.")
    parser.add_argument("--iemocap_dir", type=str, required=True, help="Path to the IEMOCAP root directory")
    
    args = parser.parse_args()

    extract_conversations(args.iemocap_dir)