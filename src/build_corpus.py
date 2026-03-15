import json
from config import DATASETS

def build_hotpotqa_corpus():
    sample_path = DATASETS["hotpotqa"]["sample_path"]
    corpus_path = DATASETS["hotpotqa"]["corpus_path"]

    seen = set()
    docs = []

    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        for title, sentences in item["context"]:
            text = " ".join(sentences).strip()
            key = title.strip()
            if key not in seen:
                seen.add(key)
                docs.append({
                    "doc_id": key,
                    "title": title,
                    "text": text
                })

    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[hotpotqa] built corpus: {len(docs)} docs")

if __name__ == "__main__":
    build_hotpotqa_corpus()