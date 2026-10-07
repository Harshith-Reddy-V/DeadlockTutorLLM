# Evaluation Framework & Benchmark

This directory contains evaluation benchmarks, ground-truth questions, and performance measurement scripts for **DeadlockTutorLLM**.

## Evaluation Objectives
Evaluate and contrast 4 system configurations:
1. **Base LLM only** (Zero-shot ungrounded LLM)
2. **Base LLM + RAG** (Retrieval-Augmented Generation)
3. **Base LLM + RAG + Fine-Tuning** (Instruction-tuned pedagogical style)
4. **Full System (LLM + RAG + Deterministic Solver)** (End-to-end DeadlockTutorLLM)

## Target Metrics
- **Numerical Accuracy**: Exact correctness on Banker's safety sequence and Need matrix calculations (Target: 100% via solver handoff).
- **Groundedness & Citation Accuracy**: Percentage of factual statements strictly supported by syllabus references.
- **Pedagogical Quality**: Step-by-step clarity, identification of student misconceptions.
- **Latency / Response Time**: Inference latency across components.
