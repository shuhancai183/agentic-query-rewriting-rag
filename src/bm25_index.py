import json
import pickle
from rank_bm25 import BM25Okapi
from config import DATASETS

def build_bm25_index(dataset_name: str):
    cfg = DATASETS[dataset_name]
    docs = []
    tokenized = []

    with open(cfg["corpus_path"], "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            docs.append(doc)
            tokenized.append(f"{doc['title']} {doc['text']}".lower().split())

    bm25 = BM25Okapi(tokenized)

    with open(cfg["bm25_path"], "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)

    print(f"[{dataset_name}] bm25 index built: {len(docs)} docs")

if __name__ == "__main__":
    for dataset_name in DATASETS.keys():
        build_bm25_index(dataset_name)