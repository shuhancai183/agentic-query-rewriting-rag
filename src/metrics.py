import re
import string
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def retrieval_recall_hotpot(retrieved_titles, gold_titles):
    retrieved_set = set(retrieved_titles)
    loose = int(len(retrieved_set & gold_titles) > 0)
    strict = int(gold_titles.issubset(retrieved_set))
    return loose, strict

def retrieval_recall_beir(retrieved_doc_ids, relevant_doc_ids):
    retrieved_set = set(retrieved_doc_ids)
    gold_set = set(relevant_doc_ids)
    loose = int(len(retrieved_set & gold_set) > 0)
    strict = int(gold_set.issubset(retrieved_set))
    return loose, strict

def hallucination_flag(prediction, docs):
    pred = normalize_answer(prediction)
    if pred == "insufficient information":
        return 0
    combined = " ".join([d["title"] + " " + d["text"] for d in docs])
    combined = normalize_answer(combined)
    return int(pred not in combined)