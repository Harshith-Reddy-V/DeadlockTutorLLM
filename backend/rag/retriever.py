"""Vector Retriever Pipeline (Modular Interface).

Supports FAISS or ChromaDB vector indices with BAAI/bge-large-en embeddings.
Full retrieval logic is implemented in Phase 3.
"""

from typing import List
from pydantic import BaseModel
from backend.rag.ingestion import DocumentChunk
from backend.config.settings import settings


class RetrievalResult(BaseModel):
    """Result of retrieving grounded chunks."""
    query: str
    chunks: List[DocumentChunk]
    has_sufficient_context: bool


class BaseRetriever:
    """Modular retriever interface."""

    def __init__(self, top_k: int = settings.top_k_retrieval):
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieves top-k relevant document chunks for the query."""
        # Phase 1 placeholder: return empty set with false sufficiency flag
        return RetrievalResult(
            query=query,
            chunks=[],
            has_sufficient_context=False
        )
