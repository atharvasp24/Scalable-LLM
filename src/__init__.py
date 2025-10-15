"""
dataset_loader.py
-----------------
Handles dataset loading, tokenization, and batching for LLM fine-tuning.
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer

def load_tokenized_dataset(model_name: str, train_file: str, val_file: str, max_length: int = 8192):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized, tokenizer
