from setuptools import setup, find_packages

setup(
    name="scalable-llm-optimization",
    version="0.1.0",
    author="Atharva Patil",
    description="Scalable LLM Training & Inference Optimization for Long-Context Retrieval using DeepSpeed, FSDP, and LoRA/QLoRA.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "deepspeed",
        "peft",
        "bitsandbytes",
        "faiss-cpu",
        "numpy",
        "pandas",
        "tqdm",
        "PyYAML",
    ],
    python_requires=">=3.8",
)
