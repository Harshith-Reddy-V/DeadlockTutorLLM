"""QLoRA Fine-Tuning Script for DeadlockTutorLLM.

This script fine-tunes a base causal language model using QLoRA 
(Quantized Low-Rank Adaptation) on the provided JSONL instruction dataset.
It exports a PEFT adapter that can be loaded into HuggingFaceProvider.
"""

import os
import argparse
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for DeadlockTutorLLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Base model ID")
    parser.add_argument("--data_path", type=str, default="finetune_data/sample.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="finetuned/adapter", help="Output directory for PEFT adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def format_prompt(sample):
    """Format the dataset rows into a prompt string for training."""
    # This formats it into ChatML or standard instruction format.
    # Qwen uses ChatML. 
    # For a general approach, we can just inject into a system/user/assistant template.
    system_msg = "You are DeadlockTutorLLM, a specialized tutor for Operating Systems deadlocks. Always teach step-by-step."
    user_msg = sample["instruction"]
    if sample.get("input"):
        user_msg += f"\nContext:\n{sample['input']}"
    assistant_msg = sample["output"]
    
    # Very basic instruction prompt format for training
    text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )
    return {"text": text}

def main():
    args = parse_args()
    
    logger.info(f"Loading dataset from {args.data_path}")
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset not found: {args.data_path}")
        return

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.map(format_prompt)

    logger.info(f"Initializing QLoRA configuration for model: {args.model_name}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # Adjust depending on model architecture
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=10,
        num_train_epochs=args.epochs,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    logger.info("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting training...")
    # NOTE: In a real environment, this can take a while.
    trainer.train() 
    
    logger.info(f"Saving PEFT adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Finished!")

if __name__ == "__main__":
    main()
