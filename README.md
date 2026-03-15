# Agentic Query Rewriting for Retrieval-Augmented Generation

Improving retrieval recall and answer accuracy in RAG pipelines via LLM-based query rewriting.

---

## Abstract

Retrieval-Augmented Generation (RAG) systems rely heavily on the quality of the initial retrieval step. Poorly formulated queries often lead to missing evidence documents, which in turn causes downstream hallucinations or incorrect answers.

This project studies whether **LLM-based query rewriting strategies** can improve retrieval performance in RAG pipelines.

We evaluate four strategies:

- Baseline (no rewriting)
- Single query rewrite
- Multi-query expansion
- Reflective rewriting

Experiments are conducted on **HotpotQA** and **BEIR-FiQA** using three retrievers:

- BM25
- Dense retrieval
- Hybrid retrieval

Results show that rewriting provides **modest but consistent improvements on multi-hop QA tasks**, while benefits are limited when dense retrieval already performs strongly.

## Research Question

Does LLM-based query rewriting improve retrieval quality and downstream QA performance in RAG pipelines?

Specifically:

1. Can rewriting improve retrieval recall?
2. Does rewriting improve answer accuracy (EM / F1)?
3. How does rewriting interact with different retrievers?
4. Does rewriting reduce hallucination in generated answers?

## Method

We implement four query rewriting strategies.

### Baseline

The original user query is directly used for retrieval.

### Single Rewrite

An LLM rewrites the query once to produce a clearer version optimized for retrieval.

### Multi-Query Rewrite

Multiple alternative rewrites are generated and combined to increase recall.

### Reflective Rewrite

An iterative rewriting approach where the model reflects on the query before generating a final reformulation.

## Experimental Setup

### Datasets

- **HotpotQA**
  - Multi-hop question answering
  - Requires retrieving multiple supporting documents

- **BEIR-FiQA**
  - Financial question answering benchmark
  - More factual and domain-specific queries

### Retrieval Models

- BM25 (sparse retrieval)
- Dense Retrieval (MiniLM embeddings)
- Hybrid Retrieval (BM25 + Dense fusion)

### Evaluation Metrics

- Retrieval Recall
- Exact Match (EM)
- F1 Score
- Hallucination Rate

## Results

### Retrieval Performance by Retriever

Dense retrieval consistently achieves the highest recall across both datasets.
![](/figures/recall_by_retriever.png)

Query rewriting provides small but measurable improvements in several configurations.
![](/figures/recall_by_dataset.png)

Key observations:

• Query rewriting provides clearer benefits on **HotpotQA**  
• Improvements are smaller on **FiQA**, where dense retrieval already performs strongly.

Hallucination rates remain relatively stable across rewriting strategies.
![](/figures/hallucination_rate.png)

## Discussion

### Rewriting helps multi-hop reasoning

HotpotQA questions often require retrieving multiple supporting documents. Query rewriting improves the chance of retrieving relevant context.

### Strong retrievers reduce rewriting gains

On BEIR-FiQA, dense retrieval already achieves high recall. In such settings, query rewriting has limited impact.

### Hybrid retrieval sensitivity

Multi-query rewriting sometimes degrades hybrid retrieval. This suggests that lexical reformulation may interfere with the sparse component of hybrid ranking.

### Key takeaway

Query rewriting is most beneficial when the baseline retriever struggles with complex or multi-hop queries.

### experiment pipeline

User Query->Query Rewriter (LLM)->Retriever(BM25 / Dense / Hybrid)->Retrieved Documents->LLM Generator->Answer->Evaluation(Recall / EM / F1 / Hallucination)

### RAG pipeline
Question->Rewrite Strategy(Baseline / Single / Multi / Reflective)->Retriever->Top-k Documents->LLM Answer Generation->Evaluation Metrics


## Contributions

This project provides:

• A reproducible experimental framework for studying query rewriting in RAG systems

• Empirical comparison across multiple datasets and retrievers

• Analysis of interactions between query rewriting and hybrid retrieval

• A modular pipeline for testing future RAG improvements
