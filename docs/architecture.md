# Architecture of DeadlockTutorLLM

DeadlockTutorLLM follows a **RAG-First Hybrid Architecture**. It combines retrieval-augmented generation with a deterministic symbolic solver and a fine-tuned LLM adapter to eliminate math hallucinations and tightly ground explanations to the course syllabus.

## Core Flow
1. **Query Routing**: The `QueryRouter` classifies student input into `THEORY`, `NUMERICAL`, `GRAPH`, or `LAB`.
2. **Knowledge Retrieval**: The `RAGRetriever` fetches syllabus context from ChromaDB/FAISS.
3. **Deterministic Solving**:
   - For `NUMERICAL` (Banker's Algorithm), the solver computes vectors step-by-step.
   - For `GRAPH` (Wait-For Graphs), the solver detects cycles via DFS.
4. **LLM Generation**: The Qwen 14B base model (with QLoRA pedagogical adapter) formulates the final text using the exact numbers from the solver.
5. **Response Composition**: The `ResponseComposer` packages the LLM explanation, solver trace, syllabus citations, and dynamically generated Mermaid graphs into the final UI payload.

## Tools & Libraries
- **Backend**: FastAPI, Pydantic, Pytest
- **AI & RAG**: HuggingFace `transformers`, `peft`, `trl`, LangChain (splitters)
- **Frontend**: Streamlit
