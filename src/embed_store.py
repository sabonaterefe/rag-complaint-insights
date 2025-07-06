import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings
from chunking import chunk_texts  # ✅ Relative import

def embed_and_store(
    input_csv="data/interim/filtered_complaints.csv",
    index_dir="vector_store/chroma_index",
    model_name="all-MiniLM-L6-v2"
):
    df = pd.read_csv(input_csv)
    texts = df["cleaned_narrative"].tolist()
    metadata = df[["Product", "Company", "Complaint ID"]].to_dict(orient="records")

    chunks = chunk_texts(texts)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)

    os.makedirs(index_dir, exist_ok=True)
    client = PersistentClient(path=index_dir)
    collection = client.get_or_create_collection(name="complaints")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[f"doc_{i}"],
            metadatas=[metadata[i % len(metadata)]]
        )

    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at {index_dir}")
