"""
fsdp_trainer.py
---------------
Implements Fully Sharded Data Parallel (FSDP) training for large LLMs.
"""

import torch
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group
from transformers import AutoModelForCausalLM

def init_fsdp_training(config_path="configs/fsdp_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.distributed.init_process_group(backend=cfg["distributed"]["backend"])

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    model = FSDP(model)

    print("✅ FSDP initialized with full sharding strategy.")
    return model
