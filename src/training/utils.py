"""
utils.py
--------
Common utilities for logging, saving models, and monitoring training.
"""

import os
import torch

def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")

def count_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total:,} | Trainable: {trainable:,}")
