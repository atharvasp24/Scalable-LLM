"""
preprocess.py
-------------
Cleans and prepares raw data for fine-tuning or retrieval.
"""

import json

def clean_text(text):
    return " ".join(text.split())

def preprocess_raw_data(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    processed = [{"text": clean_text(sample["text"])} for sample in data]
    with open(output_path, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"Preprocessed {len(processed)} samples saved to {output_path}")
