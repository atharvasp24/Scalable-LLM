"""
main.py
-------
Entry point for launching training, inference, or evaluation pipelines.
"""

from src.training.train_lora import main as train_lora
from src.inference.inference_pipeline import run_inference
from src.evaluation.benchmark_mmlu import evaluate_mmlu

if __name__ == "__main__":
    print("🚀 Scalable LLM Training & Inference Pipeline")
    print("1. Running LoRA fine-tuning...")
    train_lora()

    print("\n2. Running inference demo...")
    text = "Explain how distributed optimization improves LLM scalability."
    print(run_inference("checkpoints/lora", text))

    print("\n3. Running evaluation benchmark...")
    evaluate_mmlu("checkpoints/lora")
