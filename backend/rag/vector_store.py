"""Vector Store Implementations for DeadlockTutorLLM.

Provides:
- BaseVectorStore (Abstract base class)
- FAISSVectorStore (Native FAISS IndexFlatIP with metadata preservation)
- NumpyVectorStore (High-performance NumPy cosine similarity fallback)
- Factory function get_vector_store()
"""

import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from backend.rag.ingestion import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract interface for dense vector indices."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Adds chunks and their corresponding embeddings to the index."""
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Returns top_k (DocumentChunk, score) tuples ranked by relevance."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persists the vector index and metadata catalog to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> bool:
        """Loads index and metadata from disk. Returns True if loaded, False if not found."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns total number of chunks currently indexed."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Empties the index."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store using IndexFlatIP (Cosine similarity on normalized vectors)."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self._index = None
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            raise ImportError(
                "faiss-cpu is not installed. Run `pip install faiss-cpu` or use NumpyVectorStore."
            )

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks or not embeddings:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count must match embeddings count.")

        arr = np.array(embeddings, dtype=np.float32)
        # Normalize vectors for cosine similarity
        faiss_mod = __import__("faiss")
        faiss_mod.normalize_L2(arr)

        self._index.add(arr)
        self.chunks.extend(chunks)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        if self._index.ntotal == 0 or not self.chunks:
            return []

        q_arr = np.array([query_embedding], dtype=np.float32)
        faiss_mod = __import__("faiss")
        faiss_mod.normalize_L2(q_arr)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_arr, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        idx_file = path.with_suffix(".faiss")
        meta_file = path.with_suffix(".meta.json")

        faiss_mod = __import__("faiss")
        faiss_mod.write_index(self._index, str(idx_file))

        meta_data = [chunk.model_dump() for chunk in self.chunks]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension, "chunks": meta_data}, f, indent=2)

    def load(self, path: Path) -> bool:
        path = Path(path)
        idx_file = path.with_suffix(".faiss")
        meta_file = path.with_suffix(".meta.json")

        if not idx_file.exists() or not meta_file.exists():
            return False

        faiss_mod = __import__("faiss")
        self._index = faiss_mod.read_index(str(idx_file))
        self.dimension = self._index.d

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = [DocumentChunk(**item) for item in data.get("chunks", [])]
        return True

    def count(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def clear(self) -> None:
        self.chunks = []
        self._init_index()


class NumpyVectorStore(BaseVectorStore):
    """Pure NumPy vector store calculating cosine similarity via dot product."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks or not embeddings:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count must match embeddings count.")

        new_arr = np.array(embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(new_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_arr = new_arr / norms

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_arr
        else:
            self.embeddings = np.vstack([self.embeddings, new_arr])

        self.chunks.extend(chunks)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_arr = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, q_arr)
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(-scores)[:k]

        return [(self.chunks[idx], float(scores[idx])) for idx in top_indices]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np_file = path.with_suffix(".npy")
        meta_file = path.with_suffix(".meta.json")

        if self.embeddings is not None:
            np.save(str(np_file), self.embeddings)

        meta_data = [chunk.model_dump() for chunk in self.chunks]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension, "chunks": meta_data}, f, indent=2)

    def load(self, path: Path) -> bool:
        path = Path(path)
        np_file = path.with_suffix(".npy")
        meta_file = path.with_suffix(".meta.json")

        if not meta_file.exists():
            return False

        if np_file.exists():
            self.embeddings = np.load(str(np_file))
        else:
            self.embeddings = None

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.dimension = data.get("dimension", self.dimension)
            self.chunks = [DocumentChunk(**item) for item in data.get("chunks", [])]
        return True

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = None


def get_vector_store(dimension: int, prefer_faiss: bool = True) -> BaseVectorStore:
    """Returns FAISSVectorStore if available and requested; otherwise NumpyVectorStore."""
    if prefer_faiss:
        try:
            import faiss
            return FAISSVectorStore(dimension=dimension)
        except Exception:
            pass
    return NumpyVectorStore(dimension=dimension)
