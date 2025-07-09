def build_prompt(chunks: list[str], question: str) -> str:
    context = "\n\n".join(chunk.strip()[:500] for chunk in chunks if chunk.strip())[:1500]
    return f"""You are a helpful assistant. Use the following context to answer the question. If the context is incomplete or unclear, use your best reasoning and general knowledge to provide a helpful and informative answer.

Context:
{context}

Question: {question}

Answer:"""
