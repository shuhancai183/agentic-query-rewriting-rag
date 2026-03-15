import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import DATASETS, DENSE_MODEL_NAME, MODELS_DIR

class DenseRetriever:
    def __init__(self, dataset_name: str):
        cfg = DATASETS[dataset_name]
        self.index = faiss.read_index(cfg["dense_index_path"])
        with open(cfg["dense_meta_path"], "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.model = SentenceTransformer(DENSE_MODEL_NAME, cache_folder=MODELS_DIR)

    def search(self, query, top_k=5):
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.array(q).astype("float32")
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.docs[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": float(score),
                "source": "dense"
            })
        return results


class BM25Retriever:
    def __init__(self, dataset_name: str):
        cfg = DATASETS[dataset_name]
        with open(cfg["bm25_path"], "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.docs = data["docs"]

    def search(self, query, top_k=5):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in idxs:
            doc = self.docs[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": float(scores[idx]),
                "source": "bm25"
            })
        return results


class HybridRetriever:
    def __init__(self, dataset_name: str):
        self.dense = DenseRetriever(dataset_name)
        self.bm25 = BM25Retriever(dataset_name)

    def search(self, query, top_k=5):
        dense_results = self.dense.search(query, top_k=top_k)
        bm25_results = self.bm25.search(query, top_k=top_k)
        merged = {}

        for rank, r in enumerate(dense_results):
            key = r["doc_id"]
            merged.setdefault(key, r.copy())
            merged[key]["hybrid_score"] = merged[key].get("hybrid_score", 0) + (1.0 / (rank + 1))

        for rank, r in enumerate(bm25_results):
            key = r["doc_id"]
            if key not in merged:
                merged[key] = r.copy()
            merged[key]["hybrid_score"] = merged[key].get("hybrid_score", 0) + (1.0 / (rank + 1))

        final = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)
        for r in final:
            r["source"] = "hybrid"
        return final[:top_k]


def get_retriever(dataset_name: str, retriever_type: str):
    if retriever_type == "dense":
        return DenseRetriever(dataset_name)
    if retriever_type == "bm25":
        return BM25Retriever(dataset_name)
    if retriever_type == "hybrid":
        return HybridRetriever(dataset_name)
    raise ValueError(f"Unknown retriever_type: {retriever_type}")


def merge_results(list_of_result_lists, top_k=5):
    merged = {}
    for result_list in list_of_result_lists:
        for r in result_list:
            key = r["doc_id"]
            if key not in merged:
                merged[key] = r.copy()
                merged[key]["merge_score"] = r["score"]
            else:
                merged[key]["merge_score"] = max(merged[key]["merge_score"], r["score"])

    final = sorted(merged.values(), key=lambda x: x["merge_score"], reverse=True)
    return final[:top_k]