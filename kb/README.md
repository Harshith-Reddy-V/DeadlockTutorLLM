# Knowledge Base (KB) Instructions & Sourcing

DeadlockTutorLLM uses a local Retrieval-Augmented Generation (RAG) pipeline to ground explanations in university course materials.

---

## ⚠️ Copyright & Sourcing Policy

To comply with academic copyright and redistribution laws:
- **No copyrighted textbook PDFs are distributed in this Git repository.**
- The directory `kb/raw/` is configured in `.gitignore` to prevent committing proprietary binaries (`*.pdf`, `*.pptx`).
- Users must supply their own legally permitted course materials locally.

### Permitted Sources for Local Ingestion:
1. **University Lecture Slides**: Export your professor's lecture slides as `.pptx` or `.pdf` and place them into `kb/raw/`.
2. **Open-Access Textbooks**:
   - *Operating Systems: Three Easy Pieces (OSTEP)* by Remzi H. Arpaci-Dusseau & Andrea C. Arpaci-Dusseau. Chapters are available for free personal download at [ostep.org](https://pages.cs.wisc.edu/~remzi/OSTEP/). Download the Concurrency and Deadlock chapters directly to `kb/raw/`.
3. **Course Notes & Markdown Summaries**: Save syllabus notes or lecture transcripts as `.txt` or `.md` in `kb/raw/`.
4. **Institutional Access**: If your institution licenses *Modern Operating Systems* (Tanenbaum) or *Operating System Concepts* (Silberschatz), place authorized digital chapters in `kb/raw/`.

---

## 🛠️ How to Ingest Documents Locally

1. Place your `.pdf`, `.pptx`, `.txt`, or `.md` files in `kb/raw/`:
   ```bash
   # Example:
   # kb/raw/lecture14_deadlocks.pptx
   # kb/raw/ostep_deadlocks.pdf
   ```

2. Run the ingestion pipeline script:
   ```bash
   python scripts/ingest_kb.py
   ```

3. The script will:
   - Extract page and slide text cleanly (via `pypdf` and `python-pptx`).
   - Split content into sliding-window chunks with configurable token overlap.
   - Index chunks into the vector store (`kb/processed/` and `kb/metadata/`).

---

## 🧪 Zero-Download & Offline Testing

If no documents are placed in `kb/raw/`:
- **The deterministic solver and graph visualizer remain 100% operational.**
- All 56 unit and integration tests pass completely offline without needing any documents or model downloads.
- When asked theory questions without an indexed knowledge base, the tutor transparently informs the student that the knowledge base contains no indexed documents.
