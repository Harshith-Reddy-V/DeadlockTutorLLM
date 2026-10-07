# DeadlockTutorLLM 🔒

A domain-specific AI teaching assistant for Operating Systems (OS) deadlocks. Built from scratch for a university OS course, this project combines a deterministic symbolic solver, a local Retrieval-Augmented Generation (RAG) pipeline, and an LLM orchestration layer to teach students concepts, algorithms, graphs, and labs safely and accurately.

**Current Phase:** Phase 4 (LLM Integration and End-to-End Orchestration)

---

## 🌟 Key Features

1. **Deterministic Deadlock Solver**
   - Banker's Algorithm (Safety & Resource-Request)
   - Cycle detection for Single-Instance Wait-For Graphs (WFG)
   - Matrix-based Multi-Instance Deadlock Detection
   - *Crucially, the LLM NEVER performs matrix arithmetic. It only explains the solver's verified trace.*

2. **Course-Grounded RAG Pipeline**
   - Ingests PDFs and PPTX files directly into chunks.
   - Vector search (FAISS + BGE embeddings).
   - Strict Grounding: If the knowledge base lacks sufficient context, the tutor explicitly alerts the student that the answer is not grounded in the syllabus.

3. **Intelligent Query Router & Orchestrator**
   - Deterministically routes student queries to `THEORY`, `NUMERICAL`, `GRAPH`, or `LAB` pipelines.
   - Automatically weaves solver traces and syllabus citations into the final LLM prompt.

4. **Extensible LLM Architecture**
   - Zero-dependency `MockLLMProvider` for blazing-fast CI testing.
   - Native support for Ollama, OpenAI-Compatible (vLLM, LM Studio), and HuggingFace PEFT models.

5. **Streamlit Frontend**
   - Clean, academic interface.
   - Expanders for step-by-step solver calculations and syllabus citations.
   - Visual badges indicating query category and syllabus groundedness.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies in a virtual environment:
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the configuration template:
```bash
cp .env.example .env
```
By default, the `.env` uses the `mock` LLM provider and `mock` embedding provider. This is intentional to ensure the system boots instantly without large downloads. To use real models, edit `.env` to point to an active Ollama or LM Studio instance.

### 3. Ingesting Course Material (Optional)
Place PDFs or PPTX files into `kb/raw/` and run:
```bash
python scripts/ingest_kb.py
```

### 4. Running the Application
You must start both the backend API and the frontend UI.

**Start the FastAPI Backend (Terminal 1):**
```bash
uvicorn main:app --reload --port 8000
```

**Start the Streamlit Frontend (Terminal 2):**
```bash
streamlit run frontend/app.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Testing

The test suite runs entirely offline (using the Mock LLM and deterministic embedding logic) in seconds. It requires NO model downloads and NO API keys.

```bash
pytest tests/ -v
```
*(Currently 51/51 tests passing)*

---

## 🏗️ Architecture

Read the full architecture spec in [docs/architecture.md](docs/architecture.md).

### The Pipeline
1. **Student Input** → `/api/chat`
2. **Query Router** → Classifies as `THEORY`, `NUMERICAL`, `GRAPH`, or `LAB`.
3. **Dispatcher** → 
   - Queries RAG vector store for `THEORY` / `LAB`.
   - Executes deterministic matrices/DFS for `NUMERICAL` / `GRAPH`.
4. **Prompt Assembly** → Merges data.
5. **LLM Generation** → Explains the data.
6. **Response Composer** → Structures the output with citations and step-by-step traces.

---

## 🚧 Limitations & Future Work

- **Fine-Tuning:** The architecture contains integration points for a QLoRA fine-tuned pedagogical adapter, but the models have not yet been fine-tuned (Phase 5 planned).
- **Concurrency:** Currently optimized for local, single-user operation.
- **Vision:** Graph detection relies on manual edge inputs; future iterations could include a vision model for drawing analysis.
