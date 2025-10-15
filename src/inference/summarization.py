"""
summarization.py
----------------
Generates long-document summaries using the fine-tuned model.
"""

from src.inference.inference_pipeline import run_inference

def summarize_document(model_path, document_text):
    prompt = f"Summarize the following text in a concise way:\n\n{document_text}\n\nSummary:"
    return run_inference(model_path, prompt, max_length=2048)

if __name__ == "__main__":
    text = "Large Language Models are capable of reasoning over extended contexts..."
    summary = summarize_document("checkpoints/lora", text)
    print(summary)
