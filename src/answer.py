from openai import OpenAI
api_key=""
client =OpenAI(
            api_key=api_key,
            base_url=""
        )

def build_context(docs, max_docs=5):
    chunks = []
    for i, d in enumerate(docs[:max_docs], 1):
        chunks.append(f"[Doc {i}] {d['title']}\n{d['text']}")
    return "\n\n".join(chunks)

def answer_question(question, docs, model="gpt-4o-mini"):
    context = build_context(docs)
    prompt = f"""Answer the question using only the provided context.
If the answer cannot be found in the context, say "insufficient information".

Context:
{context}

Question: {question}

Return only the answer."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()