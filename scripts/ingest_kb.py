"""CLI script to ingest documents from kb/raw/ into vector database."""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config.settings import settings
from backend.rag.retriever import BaseRetriever


def main():
    print("=" * 60)
    print("DeadlockTutorLLM - Knowledge Base Ingestion Pipeline")
    print("=" * 60)
    print(f"Scanning raw documents directory: {settings.kb_raw_dir}")
    print(f"Target vector store path:        {settings.vector_db_path}")

    retriever = BaseRetriever()
    count = retriever.index_raw_documents()

    print("-" * 60)
    print(f"Ingestion complete: {count} document chunks indexed.")
    print(f"Processed chunks saved to: {settings.kb_processed_dir}")
    print(f"Document catalog saved to: {settings.kb_metadata_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
