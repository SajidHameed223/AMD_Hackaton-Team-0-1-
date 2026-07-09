"""
QLoRA fine-tuning for Gemma models with minimal VRAM overhead.
Logs training metrics to fine_tune_log.jsonl for efficiency tracking.
"""

import json
import os
from datetime import datetime, timezone

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset


def _log_finetune(event: dict) -> None:
    """Append training event to fine_tune_log.jsonl"""
    enriched = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open("fine_tune_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched) + "\n")


def setup_qlora_model(model_id: str, lora_r: int = 8, lora_alpha: int = 16):
    """
    Load model in 4-bit quantization with LoRA adapter.
    
    Args:
        model_id: HF model ID (e.g., "google/gemma-4-31B-it")
        lora_r: LoRA rank
        lora_alpha: LoRA scaling alpha
    
    Returns:
        model, tokenizer, peft_model
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    _log_finetune({
        "event": "setup_qlora",
        "model_id": model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    })

    return base_model, tokenizer, peft_model


def prepare_dataset(texts: list[str], tokenizer, max_length: int = 512) -> Dataset:
    """
    Tokenize texts for training.
    
    Args:
        texts: List of training text samples
        tokenizer: HF tokenizer
        max_length: Max sequence length
    
    Returns:
        HF Dataset with input_ids and attention_mask
    """
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result["labels"] = result["input_ids"].clone()
        return result

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return tokenized


def finetune(
    model_id: str,
    train_texts: list[str],
    eval_texts: list[str] = None,
    output_dir: str = "./lora_checkpoints",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    max_length: int = 512,
):
    """
    Fine-tune with QLoRA on training data.
    
    Args:
        model_id: HF model ID
        train_texts: Training samples
        eval_texts: Optional evaluation samples (for perplexity tracking)
        output_dir: Where to save LoRA weights
        num_epochs: Training epochs
        batch_size: Batch size (reduce if OOM)
        learning_rate: Learning rate for LoRA params
        max_length: Max token sequence length
    """
    print(f"[Fine-tuning] Loading model: {model_id}")
    base_model, tokenizer, peft_model = setup_qlora_model(model_id)

    print("[Fine-tuning] Preparing training dataset")
    train_dataset = prepare_dataset(train_texts, tokenizer, max_length)

    eval_dataset = None
    if eval_texts:
        print("[Fine-tuning] Preparing evaluation dataset")
        eval_dataset = prepare_dataset(eval_texts, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=len(train_dataset) // (batch_size * 2),
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=len(train_dataset) // (batch_size * 2) if eval_dataset else None,
        bf16=True,
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("[Fine-tuning] Starting training")
    trainer.train()

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    _log_finetune({
        "event": "finetune_complete",
        "model_id": model_id,
        "output_dir": output_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    })

    print(f"[Fine-tuning] LoRA weights saved to {output_dir}")
    return peft_model, tokenizer


if __name__ == "__main__":
    train_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "AMD GPUs are powerful for compute workloads.",
    ]
    eval_data = [
        "Fine-tuning improves model performance on domain tasks.",
    ]

    finetune(
        model_id="google/gemma-4-31B-it",
        train_texts=train_data,
        eval_texts=eval_data,
        num_epochs=1,
        batch_size=2,
    )
