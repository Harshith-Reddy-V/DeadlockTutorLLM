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
