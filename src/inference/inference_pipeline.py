"""
inference_pipeline.py
---------------------
Runs optimized inference using fine-tuned LoRA/QLoRA models
with long-context handling and KV cache optimization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.inference.kv_cache_optimization import enable_kv_cache
from src.inference.long_context_handler import chunk_input

def run_inference(model_path, input_text, max_length=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    enable_kv_cache(model)

    input_chunks = chunk_input(inputs, max_length)
    outputs = []

    for chunk in input_chunks:
        output_ids = model.generate(**chunk, max_new_tokens=256, do_sample=True, temperature=0.7)
        outputs.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return " ".join(outputs)

if __name__ == "__main__":
    text = "Explain how long-context retrieval improves LLM summarization accuracy."
    print(run_inference("checkpoints/lora", text))
