"""Document Ingestion & Chunking Pipeline (Interfaces & Placeholders).

Full extraction from PDFs/PPTs and chunking pipelines are scheduled for Phase 3.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Schema for a chunk of syllabus / reference text."""
    chunk_id: str
    content: str
    source: str
    page: Optional[int] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = "intermediate"
    metadata: Dict[str, Any] = Field(default_factory=dict)


def clean_text(raw_text: str) -> str:
    """Removes headers, footers, and OCR artifacts."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return " ".join(lines)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Chunks text with sliding window token/word approximation."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += step
    return chunks
