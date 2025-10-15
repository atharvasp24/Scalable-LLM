"""
retrieval_eval.py
-----------------
Evaluates retrieval accuracy and contextual relevance for long-context QA.
"""

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_retrieval(preds, labels):
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")

    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    return {"precision": precision, "recall": recall, "f1": f1}
