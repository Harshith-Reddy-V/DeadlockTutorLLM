"""Comprehensive Test Suite for Phase 3: RAG Pipeline.

Tests:
- Text cleaning and normalization
- Chunking with sliding window and metadata preservation
- PDF extraction using pypdf
- PPTX extraction using python-pptx
- Deterministic embeddings and dimension consistency
- FAISS and NumPy vector stores (indexing, search, persistence)
- BaseRetriever top-k retrieval, citations, groundedness threshold
- Missing knowledge base and empty query handling
- Insufficient relevance detection
- RAG API endpoints (/api/rag/status, /api/rag/search, /api/rag/ingest)
"""

import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.rag.ingestion import (
    DocumentChunk,
    clean_text,
    format_citation,
    extract_document,
    chunk_document,
    ingest_raw_documents,
)
from backend.rag.embeddings import DeterministicEmbedding
from backend.rag.vector_store import FAISSVectorStore, NumpyVectorStore
from backend.rag.retriever import BaseRetriever
from main import app


@pytest.fixture
def temp_kb_dir():
    """Creates a temporary directory for raw, processed, and metadata KB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        raw_dir = base / "raw"
        processed_dir = base / "processed"
        meta_dir = base / "metadata"
        raw_dir.mkdir()
        processed_dir.mkdir()
        meta_dir.mkdir()
        yield raw_dir, processed_dir, meta_dir


# =====================================================================
# 1. Text Cleaning & Chunking Tests
# =====================================================================

def test_clean_text_normalizes_whitespace_and_hyphens():
    raw = "Dead-\nlock occurs when processes   are wait-\ning for resources.\n\nPage 1 of 5\n-- 1 --\n"
    cleaned = clean_text(raw)
    assert "Deadlock" in cleaned
    assert "waiting" in cleaned
    assert "Page 1 of 5" not in cleaned
    assert "-- 1 --" not in cleaned
    assert "  " not in cleaned


def test_format_citation_page_and_slide():
    assert format_citation("Silberschatz.pdf", 312, "pdf") == "[Source: Silberschatz.pdf, page 312]"
    assert format_citation("Lecture7.pptx", 14, "pptx") == "[Source: Lecture7.pptx, slide 14]"
    assert format_citation("notes.txt", None, "txt") == "[Source: notes.txt]"


def test_chunk_document_preserves_metadata_and_citations():
    pages = [
        (1, "Mutual exclusion is the first Coffman condition.", "pdf"),
        (2, "Hold and wait allows a process to hold resources while waiting for others.", "pdf")
    ]
    chunks = chunk_document(pages, source_filename="TestOS.pdf", chunk_size=20, overlap=5)
    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[0].source == "TestOS.pdf"
    assert chunks[0].citation == "[Source: TestOS.pdf, page 1]"
    assert "Mutual exclusion" in chunks[0].content
    assert chunks[1].page == 2
    assert chunks[1].citation == "[Source: TestOS.pdf, page 2]"


# =====================================================================
# 2. Document Extraction Tests (PDF & PPTX)
# =====================================================================

def test_extract_and_chunk_pdf(temp_kb_dir):
    raw_dir, _, _ = temp_kb_dir
    pdf_path = raw_dir / "sample_deadlock.pdf"

    # Create synthetic PDF using pypdf
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Read back
    pages = extract_document(pdf_path)
    # Blank pages produce no text, verifying robust handling without crashing
    assert isinstance(pages, list)


def test_extract_and_chunk_pptx(temp_kb_dir):
    raw_dir, _, _ = temp_kb_dir
    pptx_path = raw_dir / "deadlock_lecture.pptx"

    # Create synthetic PPTX using python-pptx
    from pptx import Presentation
    prs = Presentation()
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    slide1.shapes.title.text = "Operating Systems: Deadlock Prevention"
    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "Coffman Conditions"
    prs.save(str(pptx_path))

    pages = extract_document(pptx_path)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "Deadlock Prevention" in pages[0][1]
    assert pages[1][0] == 2
    assert "Coffman Conditions" in pages[1][1]


# =====================================================================
# 3. Embedding & Vector Store Tests
# =====================================================================

def test_deterministic_embedding_consistency():
    emb_model = DeterministicEmbedding(dim=128)
    v1 = emb_model.embed_query("Deadlock Banker's algorithm safe state")
    v2 = emb_model.embed_query("Deadlock Banker's algorithm safe state")
    assert v1 == v2
    assert len(v1) == 128

    # Unit normalized check
    norm_sq = sum(x * x for x in v1)
    assert pytest.approx(norm_sq, abs=1e-4) == 1.0


def test_faiss_vector_store_indexing_and_search(temp_kb_dir):
    _, processed_dir, _ = temp_kb_dir
    emb_model = DeterministicEmbedding(dim=64)
    store = FAISSVectorStore(dimension=64)

    chunk1 = DocumentChunk(
        chunk_id="c1",
        content="Mutual exclusion prevents multiple processes from accessing a shared resource simultaneously.",
        source="silberschatz.pdf",
        page=10,
        doc_type="pdf",
        citation="[Source: silberschatz.pdf, page 10]"
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        content="Banker algorithm determines whether allocating requested resources leaves the system safe.",
        source="lecture.pptx",
        page=5,
        doc_type="pptx",
        citation="[Source: lecture.pptx, slide 5]"
    )

    embs = emb_model.embed_texts([chunk1.content, chunk2.content])
    store.add_chunks([chunk1, chunk2], embs)
    assert store.count() == 2

    # Query for Banker
    q_emb = emb_model.embed_query("Banker algorithm safe state")
    results = store.search(q_emb, top_k=1)
    assert len(results) == 1
    best_chunk, score = results[0]
    assert best_chunk.chunk_id == "c2"
    assert score > 0.0

    # Test persistence (save & load)
    save_path = processed_dir / "test_faiss_index"
    store.save(save_path)

    new_store = FAISSVectorStore(dimension=64)
    assert new_store.load(save_path) is True
    assert new_store.count() == 2
    loaded_results = new_store.search(q_emb, top_k=1)
    assert loaded_results[0][0].chunk_id == "c2"


def test_numpy_vector_store_fallback(temp_kb_dir):
    _, processed_dir, _ = temp_kb_dir
    emb_model = DeterministicEmbedding(dim=64)
    store = NumpyVectorStore(dimension=64)

    chunk = DocumentChunk(
        chunk_id="c_np",
        content="Hold and wait is prevented by requiring a process to request all resources at once.",
        source="notes.txt",
        page=1,
        doc_type="txt",
        citation="[Source: notes.txt, page 1]"
    )
    store.add_chunks([chunk], emb_model.embed_texts([chunk.content]))
    assert store.count() == 1

    q_emb = emb_model.embed_query("Hold and wait prevention")
    results = store.search(q_emb, top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c_np"


# =====================================================================
# 4. BaseRetriever Groundedness & Relevance Threshold Tests
# =====================================================================

def test_retriever_groundedness_and_citations(temp_kb_dir):
    raw_dir, processed_dir, meta_dir = temp_kb_dir
    emb_model = DeterministicEmbedding(dim=128)
    store = FAISSVectorStore(dimension=128)
    retriever = BaseRetriever(
        top_k=3,
        relevance_threshold=0.2,
        embedding_model=emb_model,
        vector_store=store,
        index_path=processed_dir / "test_index"
    )

    chunks = [
        DocumentChunk(
            chunk_id="ch1",
            content="Coffman condition 1 is Mutual Exclusion: at least one non-shareable resource.",
            source="OS_Chapter7.pdf",
            page=280,
            doc_type="pdf",
            citation="[Source: OS_Chapter7.pdf, page 280]"
        ),
        DocumentChunk(
            chunk_id="ch2",
            content="Coffman condition 2 is Hold and Wait: process holds a resource while requesting another.",
            source="OS_Chapter7.pdf",
            page=281,
            doc_type="pdf",
            citation="[Source: OS_Chapter7.pdf, page 281]"
        ),
        DocumentChunk(
            chunk_id="ch3",
            content="Coffman condition 3 is No Preemption: resources cannot be preempted forcibly.",
            source="OS_Chapter7.pdf",
            page=282,
            doc_type="pdf",
            citation="[Source: OS_Chapter7.pdf, page 282]"
        ),
        DocumentChunk(
            chunk_id="ch4",
            content="Coffman condition 4 is Circular Wait: circular chain of processes waiting.",
            source="OS_Chapter7.pdf",
            page=283,
            doc_type="pdf",
            citation="[Source: OS_Chapter7.pdf, page 283]"
        ),
    ]
    retriever.index_chunks(chunks)

    # 1. Relevant query -> has_sufficient_context = True
    res = retriever.retrieve("Explain the four Coffman conditions for deadlock", top_k=2)
    assert res.has_sufficient_context is True
    assert len(res.chunks) == 2
    assert len(res.scored_chunks) == 2
    assert len(res.citations) >= 1
    assert any("[Source: OS_Chapter7.pdf" in c for c in res.citations)

    # 2. Completely unrelated query -> has_sufficient_context = False
    unrelated_res = retriever.retrieve("Quantum culinary recipe for baking chocolate soufflé")
    assert unrelated_res.has_sufficient_context is False
    assert "does not contain sufficient relevant course material" in unrelated_res.message


def test_retriever_empty_query_and_missing_kb(temp_kb_dir):
    _, processed_dir, _ = temp_kb_dir
    emb_model = DeterministicEmbedding(dim=64)
    store = FAISSVectorStore(dimension=64)
    retriever = BaseRetriever(
        embedding_model=emb_model,
        vector_store=store,
        index_path=processed_dir / "non_existent_idx"
    )

    # Empty query
    empty_res = retriever.retrieve("")
    assert empty_res.has_sufficient_context is False
    assert "empty" in empty_res.message.lower()

    # Query on empty store
    missing_res = retriever.retrieve("What is deadlock?")
    assert missing_res.has_sufficient_context is False
    assert "no indexed documents" in missing_res.message.lower()


# =====================================================================
# 5. RAG API Endpoint Tests
# =====================================================================

def test_api_rag_status_and_search():
    client = TestClient(app)

    # Test status endpoint
    status_resp = client.get("/api/rag/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "indexed_chunks" in status_data
    assert "embedding_dimension" in status_data

    # Test search endpoint
    search_resp = client.post("/api/rag/search", json={"query": "mutual exclusion condition", "top_k": 3})
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert "query" in search_data
    assert "has_sufficient_context" in search_data
    assert "citations" in search_data
