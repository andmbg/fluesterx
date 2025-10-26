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
        """Get sentence embedding for a given text (L2-normalized)."""
        try:
            emb_raw = self.model.encode(text, normalize_embeddings=True)
            emb = np.asarray(emb_raw, dtype=float).tolist()
        except TypeError:
            emb = self.model.encode(text)
            return self._normalize(emb)
        # model returned numpy array or list
        return self._normalize(emb)

    def get_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Get embeddings for a list of texts (batched), L2-normalized."""
        try:
            embs_raw = self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True
            )
            embs = [np.asarray(e, dtype=float) for e in embs_raw]
            # ensure list of lists
            return [list(e) for e in embs]
        except TypeError:
            embs = self.model.encode(texts, batch_size=batch_size)
            return [self._normalize(e) for e in embs]

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
