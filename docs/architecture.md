# DeadlockTutorLLM Architecture

```
Student
   │
   ▼
Chat Interface (Streamlit)
   │
   ▼
FastAPI Gateway & Query Router
   │
   ├──────────────────────────┬──────────────────────────┐
   ▼                          ▼                          ▼
Theory/Concepts            Numerical Problems         Wait-for / RAG Graph
   │                          │                          │
   ▼                          ▼                          ▼
RAG Pipeline           Symbolic Solver            Graph Analysis
(BGE Embeddings +      (Banker's Algorithm,       (DFS Cycle Detection)
FAISS Vector DB)        Need/Work Matrices)              │
   │                          │                          │
   └─────────────┬────────────┴──────────────────────────┘
                 ▼
          Prompt Assembler
                 ▼
        Pedagogical LLM Engine
                 ▼
          Response Composer
                 │
                 ├─► Verified step-by-step calculations
                 ├─► Syllabus page citations
                 ├─► Clarification of common misconceptions
                 │
                 ▼
          Student Answer
```

## Guiding Principles
1. **RAG controls WHAT the system knows**: Factual content is retrieved directly from syllabus textbooks (e.g. Silberschatz) and lecture slides.
2. **Deterministic Solver guarantees NUMERICAL CORRECTNESS**: Multi-step mathematical calculations (e.g. Need matrix, Work updates, safe sequences) are never computed by the LLM. The solver performs the computation, and the LLM translates the verified trace into a student-friendly explanation.
3. **Fine-Tuning controls HOW the system teaches**: Guides the pedagogical tone—scaffolding hints, explaining errors, and separating similar concepts (e.g. *unsafe state vs. deadlock*).

---

## RAG Subsystem Architecture (Phase 3)

### 1. Ingestion Flow
```
User-Provided Documents (kb/raw/)
  ├── .pdf  (pypdf text extraction per page)
  └── .pptx (python-pptx text runs extraction per slide)
            │
            ▼
      Text Cleaning
  (hyphen reconciliation, whitespace normalization, header/footer stripping)
            │
            ▼
     Sliding-Window Chunking
  (~300-500 tokens / words with 50-word configurable overlap)
            │
            ▼
     DocumentChunk Objects
  (id, content, source, page/slide, doc_type, topic, citation)
            │
            ├─► Saved to kb/processed/chunks.json
            └─► Catalog saved to kb/metadata/catalog.json
```

### 2. Embeddings & Vector Storage
- **Embedding Layer**:
  - `DeterministicEmbedding`: Feature-hashing model generating unit-normalized dense vectors ($L_2 = 1.0$) with zero network downloads, ideal for test suites and development laptops.
  - `SentenceTransformerEmbedding`: Production wrapper around `BAAI/bge-large-en` via `sentence-transformers`.
- **Vector Database**:
  - `FAISSVectorStore`: Uses native FAISS `IndexFlatIP` (Cosine similarity on normalized vectors).
  - `NumpyVectorStore`: Seamless fallback using matrix dot products if FAISS is not present.
  - Index files persisted to `kb/processed/faiss_index.faiss` and `.meta.json`.

### 3. Retrieval & Groundedness Support
- Accepts student query, generates dense embedding vector.
- Retrieves configurable top-$k$ chunks (default: 4–6).
- Computes relevance score ($0.0 \le \text{score} \le 1.0$).
- **Groundedness Sufficiency**:
  - If $\max(\text{score}) \ge \text{relevance\_threshold}$ (default 0.25): `has_sufficient_context = True`.
  - If $\max(\text{score}) < \text{relevance\_threshold}$ or KB is empty: `has_sufficient_context = False`. The system explicitly flags that the query cannot be grounded in provided course material.

### 4. Citation Metadata
Every chunk preserves exact location metadata, formatting citations such as:
- `[Source: Silberschatz_Operating_Systems.pdf, page 318]`
- `[Source: Deadlock_Lecture_Notes.pptx, slide 14]`

### 5. Knowledge Base Usage Instructions
- **Placing documents**: Copy PDF or PPTX lecture slides into `kb/raw/`.
- **Running ingestion via CLI**:
  ```bash
  python scripts/ingest_kb.py
  ```
- **Running search via API**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/rag/search -H "Content-Type: application/json" -d "{\"query\": \"Coffman conditions\", \"top_k\": 4}"
  ```
