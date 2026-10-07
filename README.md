# DeadlockTutorLLM

> **A Domain-Specific AI Tutor for Operating System Deadlocks**  
> *Combining Retrieval-Augmented Generation (RAG), Instruction-Tuning, and Deterministic Symbolic Solvers.*

---

## 1. Problem Statement

Operating Systems deadlocks are a core undergraduate computer science concept comprising theoretical criteria (Coffman conditions, prevention vs. avoidance), multi-step algebraic algorithms (Banker's Safety & Resource-Request algorithms), and state graph models (Resource Allocation Graphs and Wait-For Graphs). 

General-purpose Large Language Models (LLMs) struggle as reliable educational tutors in this domain due to two critical issues:
1. **Numerical Hallucination**: Generative models frequently calculate matrix differences ($Need = Max - Allocation$) and multi-step resource updates ($Work = Work + Allocation$) incorrectly, leading to invalid safe sequences or false safety guarantees.
2. **Pedagogical Inconsistency & Lack of Syllabus Grounding**: Generic chat models fail to anchor their explanations to prescribed university course material and blur critical theoretical distinctions (e.g., treating an **UNSAFE state** as synonymous with **DEADLOCK**).

---

## 2. Core Architecture & Design Principles

DeadlockTutorLLM addresses these challenges with a tri-part architectural separation of concerns:

- **RAG controls WHAT the system knows**: Grounds conceptual explanations strictly in verified course slides, textbooks (e.g. Silberschatz), and faculty question banks with explicit page/slide citations.
- **Deterministic Solver guarantees NUMERICAL CORRECTNESS**: Executes all algebraic calculations (Banker's algorithm, Resource-Request, and graph cycle detection) with 100% mathematical precision.
- **Fine-Tuning controls HOW the system teaches**: Guides pedagogical behavior—enforcing step-by-step scaffolding, hint-first tutoring, misconception correction, and college examination coaching.

```
Student
   │
   ▼
Chat Interface (Streamlit)
   │
   ▼
FastAPI Gateway & Query Router
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
Theory/Concepts                Numerical Problems             Graph / Lab
   │                              │                              │
   ▼                              ▼                              ▼
RAG Pipeline                   Symbolic Solver                Cycle Analysis &
(BGE / Deterministic Embeds +  (Banker's Safety,              Concurrency Lab
FAISS Vector Store)            Resource-Request Engine)       (POSIX Threads)
   │                              │                              │
   └──────────────┬───────────────┴──────────────────────────────┘
                  ▼
           Prompt Assembler
                  ▼
        Pedagogical LLM Engine (Qwen 14B / Fallback Llama 3.1 8B)
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

---

## 3. Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- **Frontend**: Streamlit
- **Deterministic Solver**: Pure Python matrix reduction, Banker's algorithms, DFS cycle detection
- **RAG & Vector Search**: `pypdf`, `python-pptx`, native `faiss-cpu` (with NumPy dot-product fallback), `BAAI/bge-large-en` / `DeterministicEmbedding`
- **LLM Engine**: Qwen 2.5/3 14B (Primary), Llama 3.1 8B (Fallback), with Mock Provider for lightweight local CPU development
- **Testing**: Pytest

---

## 4. Project Structure

```
DeadlockTutorLLM/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # FastAPI endpoints (chat, health, solver, rag)
│   ├── solver/
│   │   ├── __init__.py
│   │   ├── models.py              # Pydantic schemas for states, traces, matrices
│   │   └── engine.py              # Deterministic deadlock algorithms
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion.py           # Document parsing (PDF/PPTX), cleaning, and chunking
│   │   ├── embeddings.py          # BGE-Large and deterministic unit-normalized embeddings
│   │   ├── vector_store.py        # FAISS IndexFlatIP & NumPy fallback vector stores
│   │   └── retriever.py           # Grounded vector retriever & citation generator
│   ├── router/
│   │   ├── __init__.py
│   │   └── query_router.py        # Intent classifier (theory, numerical, graph, lab)
│   ├── composer/
│   │   ├── __init__.py
│   │   └── response_composer.py   # Final student answer synthesizer
│   ├── llm/
│   │   ├── __init__.py
│   │   └── provider.py            # LLM abstraction (Mock, Ollama, HuggingFace)
│   └── config/
│       ├── __init__.py
│       └── settings.py            # Environment & app configuration
├── frontend/
│   └── app.py                     # Streamlit chat & tutoring interface
├── kb/
│   ├── raw/                       # Place course PDFs, lecture slides, question banks here
│   ├── processed/                 # FAISS vector database (faiss_index.faiss) & chunks.json
│   └── metadata/                  # Document catalogs (catalog.json) & source mappings
├── finetune_data/                 # Alpaca-format instruction-tuning dataset
├── evaluation/                    # Benchmarks & evaluation test suites
├── tests/                         # Pytest test suite
│   ├── test_health.py             # Health & API route tests
│   ├── test_router.py             # Query classification tests
│   ├── test_solver.py             # Exhaustive solver algorithm tests (Cases A-K)
│   ├── test_api_solver.py         # Solver HTTP endpoints tests
│   └── test_rag.py                # Document parsing, chunking, FAISS, and retrieval tests
├── scripts/
│   ├── run_dev.py                 # Multi-service development launcher
│   └── ingest_kb.py               # CLI tool to ingest raw documents into FAISS
├── docs/
│   └── architecture.md            # Detailed technical architecture design
├── requirements.txt               # Project dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules for AI/Python
├── README.md                      # Project documentation
└── main.py                        # FastAPI entrypoint
```

---

## 5. Installation Instructions

### 1. Clone the repository and checkout feature branch
```bash
git clone https://github.com/Harshith-Reddy-V/DeadlockTutorLLM.git
cd DeadlockTutorLLM
git checkout harshith-dev
```

### 2. Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

---

## 6. How to Build & Use the RAG Knowledge Base

### 1. Adding Syllabus Documents
Place course materials (textbooks, lecture slides, lab manuals) into:
```
kb/raw/
```
Supported formats: `.pdf`, `.pptx`, `.ppt`, `.txt`, `.md`.

### 2. Running Ingestion
Run the ingestion CLI script:
```bash
python scripts/ingest_kb.py
```
This extracts text, reconciles hyphens/whitespace, chunks with sliding overlap, and builds the dense vector index under `kb/processed/`.

Alternatively, trigger ingestion via the API endpoint:
```bash
curl -X POST http://127.0.0.1:8000/api/rag/ingest
```

### 3. Searching the Knowledge Base
Query relevant chunks and citations via the API:
```bash
curl -X POST http://127.0.0.1:8000/api/rag/search \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the four Coffman conditions?", "top_k": 4}'
```

---

## 7. How to Run Applications

### Run Backend (FastAPI)
```bash
python main.py
```
- API will be accessible at: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`
- RAG Status: `http://127.0.0.1:8000/api/rag/status`

### Run Frontend (Streamlit)
In a separate terminal (with `.venv` activated):
```bash
streamlit run frontend/app.py
```
- Web interface will open at: `http://localhost:8501`

---

## 8. How to Run Tests

Execute the complete test suite with Pytest:
```bash
pytest -v
```

---

## 9. Development Roadmap

- [x] **Phase 1: Project Foundation & Architecture**
- [x] **Phase 2: Deterministic Deadlock Solver** (Banker's Safety, Resource-Request, Cycle Detection, Multi-instance Reduction)
- [x] **Phase 3: RAG Knowledge Base** (PDF/PPTX ingestion, BGE/Deterministic embeddings, FAISS indexing, Citations, Groundedness)
- [ ] **Phase 4: LLM Integration** (Qwen 14B / Fallback Llama 3.1 8B, Prompt engineering, Context injection)
- [ ] **Phase 5: Query Router & Response Composer** (Pipeline orchestration)
- [ ] **Phase 6: Interactive Streamlit UI** (Trace visualization, practice mode, graph rendering)
- [ ] **Phase 7: Instruction Fine-Tuning Dataset** (1,000–3,000 verified pedagogical examples)
- [ ] **Phase 8: LoRA / QLoRA Fine-Tuning** (Subject to GPU availability)
- [ ] **Phase 9: Comprehensive Evaluation** (Benchmarked against Base LLM across 100+ verified test questions)
