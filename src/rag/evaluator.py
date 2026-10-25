"""
RAG Evaluation Pipeline
Retrieval: sentence-transformers (all-MiniLM-L6-v2)
Generation: Template-based (proxy for LLM)
Evaluation: Faithfulness, Relevance, Correctness (embedding-based proxies)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


class RAGEvaluator:
    def __init__(self, kb_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load knowledge base and embedding model.
        Model runs on CPU by default — no GPU needed.
        """
        self.model = SentenceTransformer(model_name, device="cpu")

        self.documents = []
        self.doc_embeddings = None
        self._load_knowledge_base(kb_dir)

    def _load_knowledge_base(self, kb_dir: str):
        """Load all .txt files from kb_dir as documents."""
        kb_path = Path(kb_dir)
        for f in sorted(kb_path.glob("*.txt")):
            text = f.read_text(encoding="utf-8").strip()
            if text:
                self.documents.append({"id": f.stem, "text": text})

        if self.documents:
            texts = [d["text"] for d in self.documents]
            self.doc_embeddings = self.model.encode(texts, normalize_embeddings=True)
            print(f"Knowledge base: {len(self.documents)} documents embedded.")

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text to a numpy vector (normalized)."""
        emb = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(emb).flatten()  # shape: (384,)           

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve top-k most relevant documents using cosine similarity."""
        q = self._encode(query)                    # shape: (384,)
        docs = np.asarray(self.doc_embeddings)   # shape: (N, 384)   
        sims = docs @ q                             # cosine sim (already normalized)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            {"doc": self.documents[i], "score": float(sims[i])}
            for i in top_indices
        ]

    def generate(self, query: str, context: str, mode: str = "enhanced") -> str:
        """
        Template-based generation (proxy for LLM).
        'enhanced': full context included
        'standard': truncated context + filler
        """
        if mode == "enhanced":
            return f"Based on the documentation: {context[:500]} This addresses: {query}"
        else:
            return f"Regarding your question about {query[:50]}, here is some information: {context[:100]}..."

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Proxy: embedding cosine similarity between answer and context.
        (Real RAGAS: LLM decomposes answer into claims, checks entailment.)
        """
        a = self._encode(answer)
        c = self._encode(context)
        return float(np.dot(a, c))

    def evaluate_relevance(self, question: str, answer: str) -> float:
        """
        Proxy: embedding cosine similarity between question and answer.
        (Real RAGAS: embedding similarity or LLM judge.)
        """
        q = self._encode(question)
        a = self._encode(answer)
        return float(np.dot(q, a))

    def evaluate_correctness(self, answer: str, gold_answer: str) -> float:
        """
        Proxy: embedding cosine similarity between answer and gold answer.
        (Real RAGAS: weighted blend of semantic similarity + factual check.)
        """
        a = self._encode(answer)
        g = self._encode(gold_answer)
        return float(np.dot(a, g))

    def run_evaluation(self, queries: list, gold_answers: list,
                       top_k: int = 3) -> dict:
        """
        Run full evaluation pipeline on a set of queries.

        Args:
            queries: list of question strings
            gold_answers: list of reference answers (same length)
            top_k: number of documents to retrieve

        Returns:
            dict with mean metrics for 'enhanced' and 'standard' modes
        """
        results = {"enhanced": [], "standard": []}

        for q, gold in zip(queries, gold_answers):
            retrieved = self.retrieve(q, top_k=top_k)
            context = " ".join(r["doc"]["text"] for r in retrieved)

            for mode in ["enhanced", "standard"]:
                answer = self.generate(q, context, mode=mode)
                results[mode].append({
                    "faithfulness": self.evaluate_faithfulness(answer, context),
                    "relevance": self.evaluate_relevance(q, answer),
                    "correctness": self.evaluate_correctness(answer, gold),
                })

        summary = {}
        for mode, rows in results.items():
            summary[mode] = {
                metric: np.mean([r[metric] for r in rows])
                for metric in ["faithfulness", "relevance", "correctness"]
            }
        return summary   
