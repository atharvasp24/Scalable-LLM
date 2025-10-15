"""
latency_profiler.py
-------------------
Measures end-to-end latency and throughput for inference.
"""

import time
import torch

def measure_latency(model, tokenizer, text, runs=5):
    inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    times = []
    for _ in range(runs):
        start = time.time()
        model.generate(**inputs, max_new_tokens=100)
        times.append(time.time() - start)

    avg_latency = sum(times) / len(times)
    print(f"⚡ Average Latency: {avg_latency:.3f} sec | Tokens/sec: {100 / avg_latency:.2f}")
    return avg_latency
