"""
train_lora.py
-------------
Fine-tunes LLaMA-2 or Mistral using LoRA adapters with PEFT.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from src.data.dataset_loader import load_tokenized_dataset
import json

def main():
    with open("configs/lora_config.json") as f:
        lora_cfg = json.load(f)
    with open("configs/model_config.yaml") as f:
        import yaml; model_cfg = yaml.safe_load(f)

    base_model = model_cfg["model"]["base_model"]
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")

    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)

    dataset, tokenizer = load_tokenized_dataset(
        base_model,
        model_cfg["data"]["train_file"],
        model_cfg["data"]["val_file"],
        model_cfg["model"]["max_seq_length"]
    )

    args = TrainingArguments(
        output_dir="./checkpoints/lora",
        per_device_train_batch_size=model_cfg["training"]["batch_size"],
        learning_rate=model_cfg["training"]["learning_rate"],
        num_train_epochs=model_cfg["training"]["epochs"],
        logging_steps=model_cfg["training"]["logging_steps"],
        save_steps=model_cfg["training"]["save_steps"],
        bf16=True,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"], tokenizer=tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
