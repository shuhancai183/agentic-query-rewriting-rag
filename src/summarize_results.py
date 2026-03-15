import os
import json
import csv
from collections import defaultdict
from config import RUNS_DIR, RESULTS_DIR

def summarize_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    return {
        "loose_recall": sum(x["loose_recall"] for x in data) / n,
        "strict_recall": sum(x["strict_recall"] for x in data) / n,
        "em": sum(x["em"] for x in data) / n,
        "f1": sum(x["f1"] for x in data) / n,
        "hallucination_rate": sum(x["hallucination"] for x in data) / n,
    }

def main():
    rows = []
    dataset_group = defaultdict(list)

    for filename in os.listdir(RUNS_DIR):
        if not filename.endswith(".json"):
            continue
        dataset_name, retriever_type, strategy = filename.replace(".json", "").split("__")
        metrics = summarize_file(os.path.join(RUNS_DIR, filename))
        row = {
            "dataset": dataset_name,
            "retriever": retriever_type,
            "strategy": strategy,
            **metrics
        }
        rows.append(row)
        dataset_group[dataset_name].append(row)

    summary_csv = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "retriever", "strategy", "loose_recall", "strict_recall", "em", "f1", "hallucination_rate"]
        )
        writer.writeheader()
        writer.writerows(rows)

    comparison_rows = []
    for dataset_name, ds_rows in dataset_group.items():
        best_recall = max(ds_rows, key=lambda x: x["strict_recall"])
        best_halluc = min(ds_rows, key=lambda x: x["hallucination_rate"])
        comparison_rows.append({
            "dataset": dataset_name,
            "best_strict_recall_method": f"{best_recall['retriever']} + {best_recall['strategy']}",
            "best_strict_recall": best_recall["strict_recall"],
            "lowest_hallucination_method": f"{best_halluc['retriever']} + {best_halluc['strategy']}",
            "lowest_hallucination": best_halluc["hallucination_rate"],
        })

    comparison_csv = os.path.join(RESULTS_DIR, "dataset_comparison.csv")
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "best_strict_recall_method", "best_strict_recall",
                "lowest_hallucination_method", "lowest_hallucination"
            ]
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Saved summary: {summary_csv}")
    print(f"Saved dataset comparison: {comparison_csv}")

if __name__ == "__main__":
    main()