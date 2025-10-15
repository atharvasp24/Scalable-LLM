"""
tokenizer_utils.py
------------------
Tokenizer helper utilities.
"""

from transformers import AutoTokenizer

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✅ Tokenizer loaded for {model_name}")
    return tokenizer
