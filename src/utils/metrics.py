"""
metrics.py
----------
Contains helper functions for computing loss, accuracy, and perplexity.
"""

import math
import torch

def compute_perplexity(loss):
    return round(math.exp(loss), 3)

def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return round(correct / total, 4)
