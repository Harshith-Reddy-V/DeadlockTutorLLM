"""Vector Retriever Pipeline with Groundedness Evaluation and Source Citations.

Integrates:
- Document chunk extraction & metadata
- Embedding generation
- FAISS / NumPy vector search
- Groundedness evaluation (sufficiency threshold)
- Citation formatting
"""

from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.rag.ingestion import DocumentChunk, ingest_raw_documents
from backend.rag.embeddings import BaseEmbeddingModel, get_embedding_model
from backend.rag.vector_store import BaseVectorStore, get_vector_store


class ScoredChunk(BaseModel):
    """Retrieved chunk with relevance similarity score."""
    chunk: DocumentChunk
    score: float = Field(..., description="Cosine similarity score (higher = more relevant)")
    citation: str = Field(..., description="Source citation reference")


class RetrievalResult(BaseModel):
    """Result of retrieving grounded chunks."""
    query: str
    chunks: List[DocumentChunk]
    scored_chunks: List[ScoredChunk] = Field(default_factory=list)
    has_sufficient_context: bool
    citations: List[str] = Field(default_factory=list)
    message: str


class BaseRetriever:
    """Modular RAG retriever with FAISS vector store and deterministic fallback."""

    def __init__(
        self,
        top_k: int = settings.top_k_retrieval,
        relevance_threshold: float = settings.relevance_threshold,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        index_path: Optional[Path] = None,
    ):
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.embedding_model = embedding_model or get_embedding_model()
        self.index_path = Path(index_path or settings.vector_db_path)

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            prefer_faiss = (settings.vector_db_type.lower() == "faiss")
            self.vector_store = get_vector_store(
                dimension=self.embedding_model.dimension,
                prefer_faiss=prefer_faiss
            )
            # Try loading saved index if present
            self.vector_store.load(self.index_path)

    def count(self) -> int:
        """Returns number of chunks currently in vector store."""
        return self.vector_store.count()

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embeds and indexes document chunks, then persists to disk."""
        if not chunks:
            return 0
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        self.vector_store.clear()
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save(self.index_path)
        return len(chunks)

    def index_raw_documents(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        metadata_dir: Optional[Path] = None,
    ) -> int:
        """Executes full document ingestion and indexes all resulting chunks."""
        r_dir = raw_dir or settings.kb_raw_dir
        p_dir = processed_dir or settings.kb_processed_dir
        m_dir = metadata_dir or settings.kb_metadata_dir

        chunks, report = ingest_raw_documents(
            raw_dir=r_dir,
            processed_dir=p_dir,
            metadata_dir=m_dir,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if chunks:
            self.index_chunks(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieves top-k relevant document chunks for the query with citations and grounding assessment."""
        k = top_k or self.top_k
        clean_query = query.strip()

        # Handle empty query
        if not clean_query:
            return RetrievalResult(
                query=query,
                chunks=[],
                scored_chunks=[],
                has_sufficient_context=False,
                citations=[],
                message="Query is empty.",
            )

        # If vector store is empty, try loading once
        if self.vector_store.count() == 0:
            self.vector_store.load(self.index_path)

        # If still empty: knowledge base is missing/unindexed
        if self.vector_store.count() == 0:
            return RetrievalResult(
                query=query,
                chunks=[],
                scored_chunks=[],
                has_sufficient_context=False,
                citations=[],
                message="Knowledge base contains no indexed documents. Please add documents to kb/raw/ and run ingestion.",
            )

        # Generate query embedding
        q_emb = self.embedding_model.embed_query(clean_query)
        search_results = self.vector_store.search(q_emb, top_k=k)

        scored_chunks: List[ScoredChunk] = []
        chunks: List[DocumentChunk] = []
        citations_set = set()

        for chunk, score in search_results:
            scored_chunk = ScoredChunk(
                chunk=chunk,
                score=round(score, 4),
                citation=chunk.citation,
            )
            scored_chunks.append(scored_chunk)
            chunks.append(chunk)
            citations_set.add(chunk.citation)

        # Determine groundedness sufficiency
        # Context is sufficient if at least one chunk exceeds the relevance threshold
        best_score = max((sc.score for sc in scored_chunks), default=0.0)
        has_sufficient = (best_score >= self.relevance_threshold) and len(chunks) > 0

        citations_list = sorted(list(citations_set))

        if has_sufficient:
            message = (
                f"Retrieved {len(chunks)} grounded chunks from course materials "
                f"(highest relevance score: {best_score})."
            )
        else:
            message = (
                f"The available knowledge base does not contain sufficient relevant course material to ground this query "
                f"(highest relevance score {best_score} < threshold {self.relevance_threshold})."
            )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            scored_chunks=scored_chunks,
            has_sufficient_context=has_sufficient,
            citations=citations_list,
            message=message,
        )
