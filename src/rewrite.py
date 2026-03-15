from openai import OpenAI
api_key=""
client =OpenAI(
            api_key=api_key,
            base_url=""
        )
def call_llm(prompt, model="gpt-4o-mini"):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def baseline_queries(question: str):
    return [question]

def single_rewrite_queries(question: str):
    prompt = f"""Rewrite the following question to improve document retrieval.
Keep the meaning unchanged and make it more explicit.

Question: {question}

Return only the rewritten query."""
    q = call_llm(prompt)
    return [question, q]

def multi_rewrite_queries(question: str, n=3):
    prompt = f"""Generate {n} alternative search queries for retrieving relevant documents.
Keep them concise and semantically faithful to the original question.

Question: {question}

Return one query per line, no numbering."""
    text = call_llm(prompt)
    qs = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
    return [question] + qs[:n]

def reflective_rewrite_queries(question: str, first_pass_docs):
    preview = "\n".join([f"- {d['title']}: {d['text'][:200]}" for d in first_pass_docs[:3]])
    prompt = f"""The following question did not retrieve sufficiently useful documents.

Original question:
{question}

Retrieved results preview:
{preview}

Rewrite the question so that a retriever is more likely to find the right supporting documents.
Return only one rewritten query."""
    q = call_llm(prompt)
    return [question, q]

def get_queries(strategy: str, question: str, retriever=None, top_k=5):
    if strategy == "baseline":
        return baseline_queries(question)
    if strategy == "single_rewrite":
        return single_rewrite_queries(question)
    if strategy == "multi_rewrite":
        return multi_rewrite_queries(question)
    if strategy == "reflective_rewrite":
        first_pass_docs = retriever.search(question, top_k=top_k)
        return reflective_rewrite_queries(question, first_pass_docs)
    raise ValueError(f"Unknown strategy: {strategy}")