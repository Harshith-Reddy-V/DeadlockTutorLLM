# DeadlockTutorLLM Architecture

```
Student
   │
   ▼
Chat Interface (Streamlit)
   │
   ▼
FastAPI Gateway & Chat Route (/api/chat)
   │
   ▼
Orchestrator (DeadlockTutorOrchestrator)
   │
   ├──► 1. Query Router (Regex Pattern Classifier)
   │
   ├──► 2. Dispatch
   │       ├─► THEORY / LAB ──► RAG Retriever (FAISS + BGE)
   │       └─► NUMERICAL / GRAPH ──► Deterministic Solver (Banker's, DFS Cycles)
   │
   ├──► 3. Prompt Assembly (Combines RAG context + Solver traces)
   │
   ├──► 4. LLM Provider Abstraction
   │       ├─► MockLLMProvider (for testing/offline)
   │       ├─► OllamaProvider (local models)
   │       ├─► OpenAICompatibleProvider (vLLM, LM Studio)
   │       └─► HuggingFaceProvider (transformers + PEFT)
   │
   └──► 5. Response Composer (Structured grounding, step formatting, citations)
           │
           ▼
     ChatResponse (Answer, Grounded Flag, Sources, Worked Steps, Teaching Notes)
           │
           ▼
Student (Frontend UI)
```

## Guiding Principles
1. **RAG controls WHAT the system knows**: Factual content is retrieved directly from syllabus textbooks (e.g., Silberschatz) and lecture slides. If RAG fails to find context, the system explicitly flags the response as **Not Grounded**.
2. **Deterministic Solver guarantees NUMERICAL CORRECTNESS**: Multi-step mathematical calculations (e.g., Need matrix, Work updates, safe sequences) are never computed by the LLM. The solver performs the computation, and the LLM translates the verified trace into a student-friendly explanation.
3. **Fine-Tuning controls HOW the system teaches**: Guides the pedagogical tone—scaffolding hints, explaining errors, and separating similar concepts (e.g., *unsafe state vs. deadlock*).

---

## Component Details

### 1. Orchestrator (`backend/orchestrator.py`)
The central coordinator that safely wires together the router, the RAG retriever, the deterministic solver, and the LLM. It manages prompt construction and groundedness verification, ensuring the LLM is tightly constrained by verified data.

### 2. Query Router (`backend/router/query_router.py`)
Classifies incoming student queries into:
- **THEORY**: Conceptual/definitional questions → RAG pipeline
- **NUMERICAL**: Algorithm problems with matrices/numbers → Deterministic Solver
- **GRAPH**: Wait-for graph / resource-allocation graph questions → Graph Solver
- **LAB**: Code / pthread / concurrency implementation questions → RAG + code context

### 3. Response Composer (`backend/composer/response_composer.py`)
Harmonizes all outputs. Enforces pedagogical rules (e.g., automatically appending a note explaining that an unsafe state does not guarantee deadlock for all numerical problems). Assembles structured output including `citations`, `worked_steps`, and `teaching_notes`.

### 4. LLM Providers (`backend/llm/provider.py`)
- Configured via `.env`.
- Graceful degradation ensures that if the LLM backend is offline, the pipeline still returns verified numerical results and citations using a fallback error message.
- A **MockLLMProvider** is used by default for zero-download, 100% offline automated testing.

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
```

### 2. Embeddings & Vector Storage
- **Embedding Layer**:
  - `DeterministicEmbedding`: Feature-hashing model generating unit-normalized dense vectors ($L_2 = 1.0$) with zero network downloads, ideal for test suites.
  - `SentenceTransformerEmbedding`: Production wrapper around `BAAI/bge-large-en`.
- **Vector Database**:
  - `FAISSVectorStore` (Cosine similarity on normalized vectors).
  - `NumpyVectorStore` (Fallback).

---

## Future Expansion: QLoRA Fine-Tuning
The architecture is designed to support a fine-tuned LoRA adapter injected via the `HuggingFaceProvider`.
*Note: As of Phase 4, the base implementation relies on prompt engineering and RAG/Solver grounding. The actual model weights have not yet been fine-tuned.*
