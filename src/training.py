from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from datasets import load_dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import ModelConfig

logger = logging.getLogger(__name__)


def _build_lora_config(model_cfg: ModelConfig) -> PeftLoraConfig | None:
    if model_cfg.training_params is None or model_cfg.training_params.lora is None:
        return None
    lora = model_cfg.training_params.lora
    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def _load_and_tokenize(dataset_path: str, tokenizer, max_length: int):
    ds = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(example):
        prompt = f"### Question: {example['question']}\n### Answer: {example['answer']}"
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return ds.map(tokenize, batched=False, remove_columns=ds.column_names)


def train_model(model_cfg: ModelConfig, output_dir: str = "outputs") -> Path:
    """Fine-tune a model and return the path to the saved checkpoint."""
    if model_cfg.training_params is None:
        raise ValueError(f"training_params required for model {model_cfg.name}")
    if model_cfg.dataset.train_path is None:
        raise ValueError(f"train_path required for fine-tuning model {model_cfg.name}")

    params = model_cfg.training_params
    model_output = Path(output_dir) / model_cfg.name

    logger.info("Loading model %s", model_cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_id)

    lora_config = _build_lora_config(model_cfg)
    if lora_config is not None:
        logger.info("Applying LoRA adapter (r=%d)", lora_config.r)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_ds = _load_and_tokenize(
        model_cfg.dataset.train_path, tokenizer, params.max_seq_length
    )

    training_args = TrainingArguments(
        output_dir=str(model_output),
        num_train_epochs=params.epochs,
        per_device_train_batch_size=params.batch_size,
        learning_rate=params.learning_rate,
        warmup_steps=params.warmup_steps,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        fp16=params.fp16,
        logging_steps=10,
        save_strategy="epoch",
        report_to="mlflow",
    )

    mlflow.log_params({
        "model_id": model_cfg.model_id,
        "lora_r": lora_config.r if lora_config else None,
        "epochs": params.epochs,
        "learning_rate": params.learning_rate,
        "batch_size": params.batch_size,
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting training for %s", model_cfg.name)
    trainer.train()
    trainer.save_model(str(model_output))
    tokenizer.save_pretrained(str(model_output))

    logger.info("Model saved to %s", model_output)
    return model_output
