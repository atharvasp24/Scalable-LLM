"""
retrieval_dataset.py
--------------------
Builds retrieval-aware datasets using FAISS or BM25 for long-context training.
"""

import faiss
import numpy as np
from tqdm import tqdm

class RetrievalDataset:
    def __init__(self, embeddings, docs, top_k=5):
        self.embeddings = embeddings
        self.docs = docs
        self.top_k = top_k
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query_vec):
        distances, indices = self.index.search(np.array([query_vec]), self.top_k)
        results = [self.docs[i] for i in indices[0]]
        return results
