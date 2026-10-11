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
   │       ├─► THEORY / LAB ──► RAG Retriever (FAISS + BGE / Deterministic Embeddings)
   │       └─► NUMERICAL / GRAPH ──► Deterministic Solver (Banker's, DFS Cycles, Matrix Reduction)
   │
   ├──► 3. Prompt Assembly (Combines RAG context + Solver traces)
   │
   ├──► 4. LLM Provider Abstraction
   │       ├─► MockLLMProvider (for testing/offline)
   │       ├─► OllamaProvider (local models)
   │       ├─► OpenAICompatibleProvider (vLLM, LM Studio)
   │       └─► HuggingFaceProvider (transformers + PEFT QLoRA adapter)
   │
   └──► 5. Response Composer (Structured grounding, step formatting, citations, Mermaid graph)
           │
           ▼
     ChatResponse (Answer, Grounded Flag, Sources, Worked Steps, Teaching Notes, Graph Diagram)
           │
           ▼
Student (Frontend UI: Markdown, Explanations, Dynamic Mermaid Diagrams)
```

## Guiding Principles
1. **RAG controls WHAT the system knows**: Factual content is retrieved directly from syllabus textbooks (e.g., Silberschatz, Tanenbaum, OSTEP) and lecture slides. If RAG fails to find context, the system explicitly flags the response as **Not Grounded**.
2. **Deterministic Solver guarantees NUMERICAL CORRECTNESS**: Multi-step mathematical calculations (e.g., Need matrix, Work updates, safe sequences, wait-for cycles) are never computed by the LLM. The solver performs the computation, and the LLM translates the verified trace into a student-friendly explanation.
3. **Fine-Tuning controls HOW the system teaches**: Guides the pedagogical tone—scaffolding hints, explaining errors step-by-step, and separating similar concepts (e.g., *unsafe state vs. deadlock*).
4. **Dynamic Visualization enhances UNDERSTANDING**: Resource allocation and wait-for dependencies are compiled into visual diagrams with cycles visually highlighted.

---

## Component Details

### 1. Orchestrator (`backend/orchestrator.py`)
The central coordinator that safely wires together the router, the RAG retriever, the deterministic solver, and the LLM. It manages prompt construction, system prompts, retrieval context injection, and groundedness verification, ensuring the LLM is tightly constrained by verified data.

### 2. Query Router (`backend/router/query_router.py`)
Classifies incoming student queries into:
- **THEORY**: Conceptual/definitional questions → RAG pipeline
- **NUMERICAL**: Algorithm problems with matrices/numbers → Deterministic Solver
- **GRAPH**: Wait-for graph / resource-allocation graph questions → Graph Solver & Cycle Detection
- **LAB**: Code / pthread / concurrency implementation questions → RAG + code context

### 3. Response Composer (`backend/composer/response_composer.py`)
Harmonizes all outputs. Enforces pedagogical rules (e.g., automatically appending a note explaining that an unsafe state does not guarantee deadlock for all numerical problems). Dynamically compiles wait-for graphs into Mermaid diagram syntax highlighting deadlocked cycles, and assembles structured output including `citations`, `worked_steps`, `teaching_notes`, and `graph_diagram`.

### 4. LLM Providers (`backend/llm/provider.py`)
- Configured via `.env`.
- Graceful degradation ensures that if the LLM backend is offline, the pipeline still returns verified numerical results, graph diagrams, and citations using a fallback error message.
- A **MockLLMProvider** is used by default for zero-download, 100% offline automated testing.
- A **HuggingFaceProvider** supports loading PEFT/QLoRA fine-tuned adapters seamlessly.

---

## RAG Subsystem Architecture (Phase 3 & Phase 7)

### 1. Ingestion Flow
```
User-Provided Documents (kb/raw/)
  ├── .pdf  (pypdf text extraction per page, e.g. Tanenbaum, OSTEP)
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

## Fine-Tuning Infrastructure (Phase 5)

The fine-tuning subsystem aligns the base LLM (e.g. `Qwen/Qwen2.5-14B-Instruct`) to adopt a Socratic, step-by-step tutoring methodology:
- **Training Script (`scripts/train_lora.py`)**: Uses HuggingFace `transformers`, `peft` (4-bit QLoRA with `bitsandbytes`), and `trl.SFTTrainer`.
- **Dataset Schema (`finetune_data/sample.jsonl`)**: Alpaca-style instruction format containing `instruction`, `input`, and `output` fields.
- **Dependencies (`requirements-train.txt`)**: Heavy GPU training libraries are isolated in `requirements-train.txt` to keep the base application lightweight and cross-platform.

---

## Wait-For Graph Visualization (Phase 6)

For single-instance resource systems, cycles in the Wait-For Graph correspond directly to deadlocks:
- The deterministic solver identifies back-edges using depth-first search (DFS) with three-color node marking.
- `ResponseComposer` translates graph nodes and edges into Mermaid syntax (`graph TD`).
- Edges and processes forming cycles are visually highlighted with warning styles (`fill:#f99,stroke:#333`).
- The Streamlit interface (`frontend/app.py`) renders interactive diagrams in a collapsible expander.

---

## Evaluation & Failure Resilience Framework (Phase 8)

- **Benchmark Dataset (`evaluation/dataset.json`)**: Curated test queries spanning theory, numerical algorithms, graph cycles, lab concurrency, and out-of-domain prompts, with ground-truth keywords and structured solver states.
- **Automated Evaluator (`scripts/evaluate.py`)**: Executes end-to-end queries against the orchestrator, benchmarking routing accuracy, end-to-end latency, groundedness verification, and comprehensive keyword recall across LLM text, worked steps, and solver traces.
- **Failure Resilience (`tests/test_failure_modes.py`)**: Validates defensive error handling for empty queries, out-of-domain questions, dimension mismatches, negative resource allocations, and invalid matrix claims.
