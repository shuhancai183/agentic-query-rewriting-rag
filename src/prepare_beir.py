import os
import json
import random
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from config import BEIR_DIR, DATASETS

RANDOM_SEED = 42
MAX_QUERIES = 200
MAX_CORPUS_SIZE = 6000

def resolve_beir_data_folder(base_path: str) -> str:
    direct_corpus = os.path.join(base_path, "corpus.jsonl")
    if os.path.exists(direct_corpus):
        return base_path

    for name in os.listdir(base_path):
        subdir = os.path.join(base_path, name)
        if os.path.isdir(subdir):
            sub_corpus = os.path.join(subdir, "corpus.jsonl")
            if os.path.exists(sub_corpus):
                return subdir

    raise FileNotFoundError(
        f"Could not find corpus.jsonl under {base_path}."
    )

def prepare_fiqa_subset():
    random.seed(RANDOM_SEED)

    dataset_name = "fiqa"
    out_dataset_name = "beir_fiqa"

    base_path = os.path.join(BEIR_DIR, dataset_name)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"

    os.makedirs(base_path, exist_ok=True)

    if not os.path.exists(os.path.join(base_path, "corpus.jsonl")):
        util.download_and_unzip(url, base_path)

    data_folder = resolve_beir_data_folder(base_path)
    print(f"[beir_fiqa] resolved data folder: {data_folder}")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split="test")

    selected_qids = []
    for qid in queries:
        if qid in qrels and len(qrels[qid]) > 0:
            selected_qids.append(qid)
        if len(selected_qids) >= MAX_QUERIES:
            break

    selected_doc_ids = set()
    for qid in selected_qids:
        for doc_id in qrels[qid].keys():
            selected_doc_ids.add(doc_id)

    all_doc_ids = list(corpus.keys())
    remaining_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in selected_doc_ids]
    random.shuffle(remaining_doc_ids)

    needed = max(0, MAX_CORPUS_SIZE - len(selected_doc_ids))
    selected_doc_ids.update(remaining_doc_ids[:needed])

    corpus_path = DATASETS[out_dataset_name]["corpus_path"]
    kept_corpus = {doc_id: corpus[doc_id] for doc_id in selected_doc_ids}

    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id, doc in kept_corpus.items():
            row = {
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", "")
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample_path = DATASETS[out_dataset_name]["sample_path"]
    samples = []
    for qid in selected_qids:
        rel_ids = [doc_id for doc_id in qrels[qid].keys() if doc_id in kept_corpus]
        if not rel_ids:
            continue
        samples.append({
            "qid": qid,
            "question": queries[qid],
            "relevant_doc_ids": rel_ids
        })

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[beir_fiqa] original corpus docs: {len(corpus)}")
    print(f"[beir_fiqa] subset corpus docs: {len(kept_corpus)}")
    print(f"[beir_fiqa] subset queries: {len(samples)}")
    print(f"[beir_fiqa] saved corpus to: {corpus_path}")
    print(f"[beir_fiqa] saved samples to: {sample_path}")

if __name__ == "__main__":
    prepare_fiqa_subset()