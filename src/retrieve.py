from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

def load_index(index_path="vector_store/chroma_index"):
    client = PersistentClient(path=index_path)
    return client.get_or_create_collection(name="complaints")

def embed_query(query, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode([query])[0]

def search(collection, query_embedding, top_k=5):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results

def display_results(results):
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n🔎 Match {i+1}:")
        print(doc)
        print("📎 Metadata:", results["metadatas"][0][i])

if __name__ == "__main__":
    query = "I was charged for a loan I never took"
    collection = load_index()
    embedding = embed_query(query)
    results = search(collection, embedding)
    display_results(results)
