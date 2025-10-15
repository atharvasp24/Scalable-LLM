"""
benchmark_mmlu.py
-----------------
Evaluates model performance on the MMLU benchmark.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def evaluate_mmlu(model_path, subset="college_mathematics"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    dataset = load_dataset("hendrycks_test", subset, split="test[:50]")

    correct, total = 0, 0
    for sample in tqdm(dataset):
        question = sample["question"] + "\nOptions: " + ", ".join(sample["choices"])
        inputs = tokenizer(question, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=32)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if sample["answer"].strip().lower() in answer.lower():
            correct += 1
        total += 1

    acc = round(100 * correct / total, 2)
    print(f"✅ MMLU ({subset}) Accuracy: {acc}%")
    return acc
