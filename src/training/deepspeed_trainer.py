"""
deepspeed_trainer.py
--------------------
Integrates DeepSpeed ZeRO-3 optimization for large-scale training.
"""

import deepspeed
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_deepspeed(config_path="configs/deepspeed_config.json"):
    with open(config_path) as f:
        ds_config = json.load(f)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    model_engine, _, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
    print("✅ DeepSpeed initialized with ZeRO-3 optimization.")
    return model_engine, tokenizer
