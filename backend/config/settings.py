import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file if it exists
load_dotenv(BASE_DIR / ".env")


class AppSettings(BaseModel):
    """Application configuration settings."""
    app_name: str = "DeadlockTutorLLM"
    version: str = "0.1.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    backend_host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    frontend_port: int = int(os.getenv("FRONTEND_PORT", "8501"))

    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")  # 'mock', 'ollama', 'openai_compatible'
    primary_model: str = os.getenv("PRIMARY_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    llm_api_base: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")

    # RAG Settings
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "mock")  # 'mock', 'sentence-transformers'
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en")
    vector_db_type: str = os.getenv("VECTOR_DB_TYPE", "faiss")
    vector_db_path: Path = BASE_DIR / os.getenv("VECTOR_DB_PATH", "kb/processed/faiss_index")
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # KB Paths
    kb_raw_dir: Path = BASE_DIR / os.getenv("KB_RAW_DIR", "kb/raw")
    kb_processed_dir: Path = BASE_DIR / os.getenv("KB_PROCESSED_DIR", "kb/processed")
    kb_metadata_dir: Path = BASE_DIR / os.getenv("KB_METADATA_DIR", "kb/metadata")


settings = AppSettings()
