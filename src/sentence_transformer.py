import os
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingService:
    def __init__(self, model):
        self.model = SentenceTransformer(model)

    def get_embedding(self, text: str) -> List[float]:
        """Get sentence embedding for a given text."""
        return self.model.encode(text).tolist()

    def get_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Get embeddings for a list of texts (batched)."""
        return self.model.encode(texts, batch_size=batch_size).tolist()

    def embed_transcript(
        self,
        chunks: list,
        batch: bool = True,
    ) -> dict:
        """Embed the transcript using the sentence transformer."""
        texts = [chunk["text"] for chunk in chunks]

        if batch:
            embeddings = self.get_embeddings(texts)
        else:
            embeddings = [self.get_embedding(text) for text in texts]

        for i, emb in enumerate(embeddings):
            chunks[i]["embedding"] = emb

        del texts
        del embeddings

        return chunks
