# Fine-Tuning Data

This directory stores datasets used to train the QLoRA adapter for the LLM.

## Purpose
The primary objective of fine-tuning is **not** to teach the LLM factual OS knowledge (RAG handles that). Instead, fine-tuning aligns the model's **teaching style**. It trains the model to:
1. Explain step-by-step.
2. Provide Socratic hints instead of direct answers when requested.
3. Obey the numbers output by the deterministic solver rather than inventing math.

## Files
- `sample.jsonl`: A small illustrative set of instruction-input-output examples demonstrating the expected schema. 
  > ⚠️ **Note**: This sample file is intended exclusively for pipeline validation and schema testing. It is **not** sufficient in size or diversity for production fine-tuning (which typically requires 1,000–3,000 curated examples).
- `scripts/train_lora.py`: The QLoRA training script using the `trl` and `peft` libraries to generate the final adapter.
