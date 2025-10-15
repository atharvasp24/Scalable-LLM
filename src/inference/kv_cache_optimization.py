"""
kv_cache_optimization.py
------------------------
Implements KV-cache reuse for faster long-sequence inference.
"""

import torch

def enable_kv_cache(model):
    for module in model.modules():
        if hasattr(module, "use_cache"):
            module.use_cache = True
    torch.cuda.empty_cache()
    print("✅ KV Cache optimization enabled for inference.")
