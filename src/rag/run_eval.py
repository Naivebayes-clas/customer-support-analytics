"""Run RAG evaluation on the knowledge base."""
import json
import numpy as np
from src.rag.evaluator import RAGEvaluator

# Load Q&A pairs
with open("./data/qa_pairs_ravenstack.json") as f:
    qa_pairs = json.load(f)

queries = [p["query"] for p in qa_pairs]
gold_answers = [p["gold_answer"] for p in qa_pairs]
expected_docs = [p["expected_doc"] for p in qa_pairs]

print(f"Loaded {len(qa_pairs)} Q&A pairs")
print(f"Knowledge base: ./data/knowledge_base/")
print()

evaluator = RAGEvaluator("./data/knowledge_base_ravenstack")

# Retrieval accuracy (top-1)
correct = 0
for q, expected in zip(queries, expected_docs):
    retrieved = evaluator.retrieve(q, top_k=1)
    if retrieved[0]["doc"]["id"] == expected:
        correct += 1
retrieval_acc = correct / len(queries)
print(f"Retrieval accuracy (top-1): {retrieval_acc:.3f} ({correct}/{len(queries)})")
print()

# Full evaluation
results = evaluator.run_evaluation(queries, gold_answers, top_k=3)

print("=== RAG Evaluation Results ===")
for mode, metrics in results.items():
    print(f"\n{mode.upper()} mode:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")   
