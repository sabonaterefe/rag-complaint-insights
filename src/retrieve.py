from chromadb import PersistentClient
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.embedding_wrapper import ChromaCompatibleEmbeddingFunction

def load_vectorstore():
    client = PersistentClient(path="chroma_db")
    embedding_fn = ChromaCompatibleEmbeddingFunction()
    return client.get_or_create_collection(name="complaints", embedding_function=embedding_fn)

def retrieve_top_k_chunks(query: str, k: int = 3):
    collection = load_vectorstore()
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0], results["metadatas"][0]
