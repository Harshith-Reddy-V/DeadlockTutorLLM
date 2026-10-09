# Evaluation Framework

This directory contains the testing harness for Phase 8 of DeadlockTutorLLM.

## Components
- `dataset.json`: A manually curated test set covering Theory, Numerical, Graph, Lab, and Out-of-Domain questions.
- `evaluate.py`: An automated script designed to run queries through the system and benchmark keyword accuracy and latency.

## Metrics
1. **Factual Correctness**: Measured via keyword recall against the ground truth.
2. **Numerical Grounding**: Ensured programmatically because the LLM is explicitly given the trace from the Deterministic Solver.
3. **Resilience**: Evaluated via `tests/test_failure_modes.py` to handle empty strings, matrix mismatches, and negative requests gracefully.
