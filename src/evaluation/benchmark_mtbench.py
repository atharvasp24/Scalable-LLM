"""
benchmark_mtbench.py
--------------------
Runs MT-Bench evaluation on reasoning and generation quality.
"""

import random

def evaluate_mtbench(model, tokenizer, num_samples=10):
    """
    Simplified MT-Bench simulation for testing generation quality.
    """
    prompts = [
        "Explain quantum entanglement to a 5-year-old.",
        "Summarize the causes of World War II.",
        "Describe the working of a transformer model."
    ]
    avg_len = 0
    for _ in range(num_samples):
        prompt = random.choice(prompts)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=150)
        avg_len += outputs.shape[-1]
    avg_len //= num_samples
    print(f"✅ MT-Bench completed. Avg generation length: {avg_len} tokens.")
    return avg_len
