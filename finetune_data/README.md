# Fine-Tuning Data

This directory stores datasets used to train the QLoRA adapter for the LLM.

## Purpose
The primary objective of fine-tuning is **not** to teach the LLM factual OS knowledge (RAG handles that). Instead, fine-tuning aligns the model's **teaching style**. It trains the model to:
1. Explain step-by-step.
2. Provide Socratic hints instead of direct answers when requested.
3. Obey the numbers output by the deterministic solver rather than inventing math.

## Files
- `sample.jsonl`: A curated set of instruction-input-output pairs formatted for causal language modeling.
- `scripts/train_lora.py`: The QLoRA training script using the `trl` and `peft` libraries to generate the final adapter.
