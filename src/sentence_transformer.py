"""Sentence Transformer embedding service."""

from typing import List

import numpy as np
from torch import Tensor
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrapper around SentenceTransformer for embedding texts.

    Mainly provides the `embed_transcript()` method to embed transcript chunks.
    The other methods wrap around SentenceTransformer methods.
    """

    def __init__(self, model):
        self.model = SentenceTransformer(model)

    def _normalize(self, emb: List[float] | Tensor) -> List[float]:
        a = np.asarray(emb, dtype=float)
        n = np.linalg.norm(a) + 1e-12
        return (a / n).tolist()

    def get_embedding(self, text: str) -> List[float]:
        emb = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=float).tolist()

    def get_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        embs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )
        return [np.asarray(e, dtype=float).tolist() for e in embs]

    def embed_transcript(
        self,
        chunks: list,
        batch: bool = True,
    ) -> list:
        """Embed the transcript using the sentence transformer and store L2-normalized vectors."""
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
