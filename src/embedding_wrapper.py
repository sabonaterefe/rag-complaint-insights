from sentence_transformers import SentenceTransformer

class ChromaCompatibleEmbeddingFunction:
    def __init__(self, model_name="all-mpnet-base-v2"):  # 🔁 Upgraded model
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, normalize_embeddings=True).tolist()

    def name(self) -> str:
        return self.model_name
