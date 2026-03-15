# Agentic Query Rewriting for Retrieval-Augmented Generation

Improving retrieval recall and answer accuracy in Retrieval-Augmented Generation (RAG) pipelines via LLM-based query rewriting.

---

# Abstract

Retrieval-Augmented Generation (RAG) systems rely heavily on the quality of the retrieval step. Poorly formulated queries can lead to missing evidence documents, which often results in hallucinated or incorrect answers from downstream language models.

This project investigates whether **LLM-based query rewriting strategies** can improve retrieval quality and downstream QA performance in RAG pipelines.

We evaluate four query strategies:

- Baseline (no rewriting)
- Single query rewrite
- Multi-query expansion
- Reflective rewriting

Experiments are conducted on two datasets:

- **HotpotQA** (multi-hop QA)
- **BEIR-FiQA** (financial retrieval benchmark)

and three retrievers:

- BM25
- Dense Retrieval (MiniLM embeddings)
- Hybrid Retrieval (BM25 + Dense)

Our results show that query rewriting provides **consistent improvements on multi-hop QA tasks**, while benefits are limited when strong dense retrievers already achieve high recall.

---

# Research Question

Does LLM-based query rewriting improve retrieval quality and downstream QA performance in RAG pipelines?

Specifically:

1. Can rewriting improve retrieval recall?
2. Does rewriting improve answer accuracy (EM / F1)?
3. How does rewriting interact with different retrievers?
4. Does rewriting reduce hallucination in generated answers?

---

# Method

We implement four query rewriting strategies.

## Baseline

The original user query is directly used for retrieval without modification.

## Single Rewrite

A language model rewrites the query once to produce a clearer retrieval-oriented formulation.

## Multi-Query Rewrite

Multiple alternative rewrites are generated and used jointly for retrieval to increase coverage of relevant documents.

## Reflective Rewrite

The model first analyzes potential issues in the query and then generates a refined query.

---

# Experimental Setup

## Datasets

### HotpotQA

Multi-hop question answering benchmark.

- Requires retrieving **multiple supporting documents**
- Evaluates both retrieval quality and reasoning ability

### BEIR-FiQA

Financial question answering dataset from the BEIR benchmark.

- Domain-specific financial queries
- Primarily tests factual retrieval

---

## Retrieval Models

Three retrieval systems are evaluated:

BM25

Sparse lexical retrieval.

Dense Retrieval

MiniLM embedding-based semantic retrieval.

Hybrid Retrieval

Fusion of BM25 and dense scores.

---

## Evaluation Metrics

Retrieval Metrics

- Loose Recall
- Strict Recall

Generation Metrics

- Exact Match (EM)
- F1 Score

Reliability

- Hallucination Rate

---

# Results

## Retrieval Recall

Dense retrieval achieves the highest recall across both datasets.
![](/figures/recall_by_retriever.png)

However, query rewriting provides consistent improvements for weaker retrievers and complex queries.
![](/figures/recall_by_dataset.png)

---

## BEIR-FiQA

Dense retrieval already performs strongly.

Retriever | Strategy | Loose Recall
--- | --- | ---
Dense | Baseline | 0.855
Dense | Single Rewrite | 0.845
Dense | Multi Rewrite | 0.840
Dense | Reflective Rewrite | 0.840

Because dense retrieval already captures semantic similarity well, rewriting produces minimal improvements.

For BM25:

Retriever | Strategy | Loose Recall
--- | --- | ---
BM25 | Baseline | 0.455
BM25 | Multi Rewrite | 0.510

Multi-query rewriting improves recall by ~12% relative gain.

---

## HotpotQA

HotpotQA requires retrieving multiple supporting documents.

Strategy | Loose Recall
--- | ---
Baseline | 0.778
Multi Rewrite | 0.792

For dense retrieval:

Strategy | Loose Recall
--- | ---
Baseline | 0.798
Multi Rewrite | 0.802

Reflective rewriting achieves the highest strict recall:

Strategy | Strict Recall
--- | ---
Reflective Rewrite | 0.260

---

# QA Performance

On HotpotQA, rewriting also improves answer quality.

Retriever | Strategy | EM | F1
--- | --- | --- | ---
BM25 | Reflective Rewrite | 0.266 | 0.335
Dense | Reflective Rewrite | 0.260 | 0.340

Compared to baseline:

EM improves from 0.236 → 0.260  
F1 improves from 0.316 → 0.340

This indicates that improved retrieval can translate into better answer generation.

---

# Hallucination Rate

Hallucination rates remain relatively stable.
![](/figures/hallucination_rate.png)

Dataset | Retriever | Strategy | Hallucination
--- | --- | --- | ---
HotpotQA | BM25 | Baseline | 0.066
HotpotQA | BM25 | Reflective Rewrite | 0.054

Reflective rewriting slightly reduces hallucination in some settings.

---

# Hybrid Retrieval Behavior

Hybrid retrieval shows sensitivity to query rewriting.

Example on BEIR-FiQA:

Strategy | Loose Recall
--- | ---
Baseline | 0.805
Multi Rewrite | 0.305

Multi-query rewriting significantly degrades hybrid performance.

This suggests lexical expansion may conflict with sparse ranking signals.

---

# Discussion

## Rewriting Helps Multi-Hop Retrieval

HotpotQA questions often require retrieving multiple supporting passages.

Query rewriting increases the chance of retrieving relevant documents for complex queries.

---

## Strong Dense Retrievers Reduce Rewriting Gains

On BEIR-FiQA, dense retrieval already achieves high recall (0.855).

In such cases, rewriting has limited additional benefit.

---

## Hybrid Retrieval is Sensitive to Reformulation

Hybrid systems rely partly on lexical matching.

Aggressive rewriting may remove important lexical cues, degrading sparse retrieval scores.

---

## Key Takeaway

Query rewriting is most beneficial when:

- the retriever struggles with lexical mismatch
- queries require multi-hop reasoning
- sparse retrievers are used

---

# Experiment Pipeline

User Query
→ Query Rewriter (LLM)
→ Retriever (BM25 / Dense / Hybrid)
→ Retrieved Documents
→ LLM Answer Generation
→ Evaluation (Recall / EM / F1 / Hallucination)

---

# RAG Pipeline

Question
→ Rewrite Strategy (Baseline / Single / Multi / Reflective)
→ Retriever
→ Top-k Documents
→ LLM Answer Generation
→ Evaluation Metrics

---

# Reproducing the Experiments

Install dependencies

pip install -r requirements.txt

---

Dataset Preparation

python src/prepare_hotpotqa.py  
python src/prepare_beir.py

---

Run Full Pipeline

python src/prepare_hotpotqa.py  
python src/build_corpus.py  
python src/prepare_beir.py  
python src/dense_index.py  
python src/bm25_index.py  
python src/runner.py  
python src/summarize_results.py  
python src/plot_results.py  

Outputs:

results/  → experiment metrics  
figures/  → generated plots  

---

# Project Structure

agentic-query-rewriting-rag

src  
prepare_hotpotqa.py  
build_corpus.py  
prepare_beir.py  
dense_index.py  
bm25_index.py  
runner.py  
summarize_results.py  
plot_results.py  

results  

figures  

README.md  
requirements.txt  

---

# Contributions

This project provides:

- A reproducible experimental framework for studying query rewriting in RAG systems
- Empirical comparison across multiple datasets and retrievers
- Analysis of interactions between query rewriting and hybrid retrieval
- A modular pipeline for testing future RAG improvements
