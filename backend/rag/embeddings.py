"""Modular Embedding Models for DeadlockTutorLLM RAG.

Provides:
- BaseEmbeddingModel (Abstract base class)
- DeterministicEmbedding (Lightweight, zero-download, fast test/CPU embedding)
- SentenceTransformerEmbedding (Production BAAI/bge-large-en wrapper via sentence-transformers)
- Factory function get_embedding_model()
"""

import math
import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Optional
from backend.config.settings import settings


class BaseEmbeddingModel(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of text documents."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimension."""
        pass


class DeterministicEmbedding(BaseEmbeddingModel):
    """Lightweight, deterministic embedding provider based on feature hashing.

    Requires NO large downloads, PyTorch, or GPU.
    Generates unit-normalized (L2 norm = 1.0) dense representations.
    Inner product reflects token overlap, n-gram semantics, and term frequency.
    Ideal for unit tests, development, and resource-constrained environments.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self._dim

    def _embed_single(self, text: str) -> List[float]:
        vector = [0.0] * self._dim
        clean = text.lower()
        tokens = re.findall(r"\b\w+\b", clean)
        if not tokens:
            return vector

        # Bag of words and character bigrams for subword robustness
        for token in tokens:
            idx = self._hash_token(token)
            vector[idx] += 1.0

        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            idx = self._hash_token(bigram)
            vector[idx] += 0.5

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-9:
            vector = [v / norm for v in vector]
        return vector

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(query)


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Production embedding model using HuggingFace / sentence-transformers (e.g. BAAI/bge-large-en)."""

    def __init__(self, model_name: str = "BAAI/bge-large-en"):
        self.model_name = model_name
        self._model = None
        self._dimension = 1024 if "large" in model_name else 384

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. Run `pip install sentence-transformers` "
                    "or set EMBEDDING_PROVIDER=mock in your .env to use the lightweight deterministic embedding."
                )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        self._load_model()
        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()


def get_embedding_model(provider: Optional[str] = None) -> BaseEmbeddingModel:
    """Factory to retrieve configured embedding provider."""
    prov = (provider or getattr(settings, "embedding_provider", "mock")).lower()
    if prov in ("sentence-transformers", "bge", "bge-large", "hf"):
        try:
            return SentenceTransformerEmbedding(model_name=settings.embedding_model)
        except Exception:
            # Graceful fallback to deterministic mock if sentence-transformers not ready
            return DeterministicEmbedding(dim=384)
    return DeterministicEmbedding(dim=384)
