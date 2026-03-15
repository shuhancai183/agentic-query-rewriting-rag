import json
import itertools
import os
from config import DATASETS, RUNS_DIR
from retriever import get_retriever, merge_results
from rewrite import get_queries
from answer import answer_question
from metrics import (
    exact_match_score,
    f1_score,
    retrieval_recall_hotpot,
    retrieval_recall_beir,
    hallucination_flag,
)

TOP_K = 5

DATASET_LIST = ["hotpotqa", "beir_fiqa"]
RETRIEVERS = ["dense", "bm25", "hybrid"]
STRATEGIES = ["baseline", "single_rewrite", "multi_rewrite", "reflective_rewrite"]

def load_dataset_items(dataset_name):
    sample_path = DATASETS[dataset_name]["sample_path"]
    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_hotpot(item, retrieved):
    q = item["question"]
    gold_answer = item["answer"]
    gold_titles = {t for t, _ in item["supporting_facts"]}
    pred = answer_question(q, retrieved)
    retrieved_titles = [d["title"] for d in retrieved]
    loose_rec, strict_rec = retrieval_recall_hotpot(retrieved_titles, gold_titles)

    return {
        "question": q,
        "gold_answer": gold_answer,
        "pred_answer": pred,
        "retrieved_ids_or_titles": retrieved_titles,
        "loose_recall": loose_rec,
        "strict_recall": strict_rec,
        "em": exact_match_score(pred, gold_answer),
        "f1": f1_score(pred, gold_answer),
        "hallucination": hallucination_flag(pred, retrieved),
    }

def evaluate_beir(item, retrieved):
    q = item["question"]
    relevant_doc_ids = item["relevant_doc_ids"]
    retrieved_doc_ids = [d["doc_id"] for d in retrieved]
    loose_rec, strict_rec = retrieval_recall_beir(retrieved_doc_ids, relevant_doc_ids)

    pred = answer_question(q, retrieved)
    return {
        "question": q,
        "gold_answer": "",
        "pred_answer": pred,
        "retrieved_ids_or_titles": retrieved_doc_ids,
        "loose_recall": loose_rec,
        "strict_recall": strict_rec,
        "em": 0,
        "f1": 0,
        "hallucination": hallucination_flag(pred, retrieved),
    }

def run_one(dataset_name, retriever_type, strategy):
    print(f"\n=== Running {dataset_name} | {retriever_type} | {strategy} ===")
    retriever = get_retriever(dataset_name, retriever_type)
    data = load_dataset_items(dataset_name)

    outputs = []
    for idx, item in enumerate(data):
        q = item["question"]
        queries = get_queries(strategy, q, retriever=retriever, top_k=TOP_K)
        all_results = [retriever.search(query, top_k=TOP_K) for query in queries]
        retrieved = merge_results(all_results, top_k=TOP_K)

        if dataset_name == "hotpotqa":
            row = evaluate_hotpot(item, retrieved)
        else:
            row = evaluate_beir(item, retrieved)

        outputs.append(row)

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx+1}/{len(data)}")

    out_name = f"{dataset_name}__{retriever_type}__{strategy}.json"
    out_path = os.path.join(RUNS_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")

def main():
    for dataset_name, retriever_type, strategy in itertools.product(
        DATASET_LIST, RETRIEVERS, STRATEGIES
    ):
        run_one(dataset_name, retriever_type, strategy)

if __name__ == "__main__":
    main()