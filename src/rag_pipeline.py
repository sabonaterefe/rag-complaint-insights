from src.retrieve import retrieve_top_k_chunks
from src.prompt_template import build_prompt
from src.generator import generate_answer
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def is_context_relevant(question: str, chunks: list[str], threshold: float = 0.3) -> bool:
    q_emb = embedding_model.encode(question, convert_to_tensor=True)
    for chunk in chunks:
        if chunk.strip():
            c_emb = embedding_model.encode(chunk, convert_to_tensor=True)
            score = float(util.cos_sim(q_emb, c_emb)[0][0])
            if score >= threshold:
                return True
    return False

def build_rag_chain():
    def rag_chain(question: str, model: str = "gemma:2b", k: int = 5):
        chunks, metadatas = retrieve_top_k_chunks(question, k=k)

        # 🔍 Check if context is semantically relevant
        if not chunks or not is_context_relevant(question, chunks):
            fallback_prompt = f"""You are a helpful assistant. Answer the following question using your general knowledge and reasoning:

Question: {question}

Answer:"""
            answer = generate_answer(fallback_prompt, model)
            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": [],
                "metadatas": []
            }

        # ✅ Use RAG prompt with context
        prompt = build_prompt(chunks, question)
        answer = generate_answer(prompt, model)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks,
            "metadatas": metadatas
        }

    return rag_chain
