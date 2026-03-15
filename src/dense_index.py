import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import DATASETS, DENSE_MODEL_NAME, MODELS_DIR

def build_dense_index(dataset_name: str):
    cfg = DATASETS[dataset_name]
    docs = []

    with open(cfg["corpus_path"], "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    texts = [f"{d['title']}. {d['text']}" for d in docs]

    model = SentenceTransformer(DENSE_MODEL_NAME, cache_folder=MODELS_DIR)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, cfg["dense_index_path"])

    with open(cfg["dense_meta_path"], "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print(f"[{dataset_name}] dense index built: {len(docs)} docs")

if __name__ == "__main__":
    for dataset_name in DATASETS.keys():
        build_dense_index(dataset_name)