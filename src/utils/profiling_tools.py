"""
profiling_tools.py
------------------
Profiling utilities to measure GPU memory and runtime.
"""

import torch
import time

def profile_runtime(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"⏱ Runtime: {end - start:.2f}s")
    return result

def profile_memory():
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"💾 GPU Memory Used: {mem:.2f} GB")
