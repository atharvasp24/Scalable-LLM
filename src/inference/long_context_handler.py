"""
long_context_handler.py
-----------------------
Handles long-context splitting and positional interpolation.
"""

def chunk_input(inputs, max_length=4096):
    """
    Splits tokenized input into smaller chunks for long-context generation.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    num_chunks = (input_ids.shape[1] + max_length - 1) // max_length
    chunks = []

    for i in range(num_chunks):
        start = i * max_length
        end = min((i + 1) * max_length, input_ids.shape[1])
        chunks.append({
            "input_ids": input_ids[:, start:end],
            "attention_mask": attention_mask[:, start:end],
        })

    print(f"📄 Input split into {len(chunks)} chunks for long-context processing.")
    return chunks
