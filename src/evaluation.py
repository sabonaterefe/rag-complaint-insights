import pandas as pd
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.rag_pipeline import build_rag_chain


semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_answer(generated: str, reference: str, method: str = "semantic") -> float:
    """
    Compare generated and reference answers using semantic or exact match.
    """
    if not generated or not reference or reference == "N/A":
        return 0.0

    if method == "exact":
        return SequenceMatcher(None, generated.lower(), reference.lower()).ratio()

    elif method == "semantic":
        emb1 = semantic_model.encode(generated, convert_to_tensor=True)
        emb2 = semantic_model.encode(reference, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])

    else:
        raise ValueError("Unsupported evaluation method")

def run_evaluation(
    questions: list[str],
    output_path: str,
    method: str = "semantic",
    threshold: float = 0.75,
    model: str = "gemma:2b"
):
    """
    Run RAG pipeline on a list of questions and save evaluation results.
    """
    rag_chain = build_rag_chain()
    results = []

    for question in questions:
        print(f"🔍 Evaluating: {question}")
        result = rag_chain(question, model=model, k=3)
        answer = result["answer"]
        reference = "N/A"  # Placeholder — replace with real references if available
        score = evaluate_answer(answer, reference, method) if reference != "N/A" else None
        is_weak = score is not None and score < threshold

        results.append({
            "question": question,
            "generated_answer": answer,
            "similarity_score": score,
            "is_weak": is_weak,
            "retrieved_chunks": "\n---\n".join(result["retrieved_chunks"])
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ Evaluation results saved to {output_path}")
