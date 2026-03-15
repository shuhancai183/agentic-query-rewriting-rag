import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpora")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
BEIR_DIR = os.path.join(DATA_DIR, "beir")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BEIR_DIR, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATASETS = {
    "hotpotqa": {
        "type": "qa",
        "sample_path": os.path.join(DATA_DIR, "hotpotqa_dev_sample.json"),
        "corpus_path": os.path.join(CORPUS_DIR, "hotpotqa_corpus.jsonl"),
        "dense_index_path": os.path.join(INDEX_DIR, "hotpotqa_dense.index"),
        "dense_meta_path": os.path.join(INDEX_DIR, "hotpotqa_dense_meta.json"),
        "bm25_path": os.path.join(INDEX_DIR, "hotpotqa_bm25.pkl"),
    },
    "beir_fiqa": {
        "type": "ir",
        "sample_path": os.path.join(DATA_DIR, "beir_fiqa_test.json"),
        "corpus_path": os.path.join(CORPUS_DIR, "beir_fiqa_corpus.jsonl"),
        "dense_index_path": os.path.join(INDEX_DIR, "beir_fiqa_dense.index"),
        "dense_meta_path": os.path.join(INDEX_DIR, "beir_fiqa_dense_meta.json"),
        "bm25_path": os.path.join(INDEX_DIR, "beir_fiqa_bm25.pkl"),
    }
}