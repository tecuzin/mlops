"""Training worker — polls the API for pending finetune runs and processes them."""
from __future__ import annotations

import logging
import os
import time
import traceback

import httpx
import mlflow
from datasets import load_dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://api:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")


def _notify(run_id: int, status: str, logs: str = "", metrics: dict | None = None, error: str = ""):
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        client.patch(f"/runs/{run_id}/status", params={"status": status})
        if logs:
            client.patch(f"/runs/{run_id}/logs", params={"text": logs})
        if metrics:
            client.post(f"/runs/{run_id}/results", json=metrics)
        if error:
            client.patch(f"/runs/{run_id}/logs", params={"text": f"ERROR: {error}"})


def _build_lora_config(params: dict) -> PeftLoraConfig | None:
    lora = params.get("lora")
    if not lora:
        return None
    return PeftLoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        target_modules=lora["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def process_run(run: dict) -> None:
    run_id = run["id"]
    config = run["config_snapshot"]
    params = run.get("training_params") or config.get("training_params", {})
    model_hf_id = config["model_id"]

    logger.info("Processing training run %d — model %s", run_id, model_hf_id)
    _notify(run_id, "training", logs=f"Démarrage de l'entraînement pour {model_hf_id}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.get("experiment_name", "mlops-default"))

    with mlflow.start_run(run_name=config["model_name"]) as mlrun:
        _notify(run_id, "training", logs=f"MLflow run ID: {mlrun.info.run_id}")

        with httpx.Client(base_url=API_URL, timeout=10) as client:
            client.patch(f"/runs/{run_id}/status", params={"status": "training"})

        tokenizer = AutoTokenizer.from_pretrained(model_hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_hf_id)
        lora_config = _build_lora_config(params)
        if lora_config:
            model = get_peft_model(model, lora_config)
            _notify(run_id, "training", logs=f"LoRA appliqué (r={lora_config.r})")

        train_ds_path = _resolve_dataset_path(run, "train")
        ds = load_dataset("json", data_files=train_ds_path, split="train")

        max_len = params.get("max_seq_length", 512)

        def tokenize(example):
            prompt = f"### Question: {example['question']}\n### Answer: {example['answer']}"
            return tokenizer(prompt, truncation=True, max_length=max_len, padding="max_length")

        train_ds = ds.map(tokenize, batched=False, remove_columns=ds.column_names)

        output_path = f"{OUTPUT_DIR}/{config['model_name']}"
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=params.get("epochs", 3),
            per_device_train_batch_size=params.get("batch_size", 4),
            learning_rate=params.get("learning_rate", 2e-5),
            warmup_steps=params.get("warmup_steps", 100),
            gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
            fp16=params.get("fp16", True),
            logging_steps=10,
            save_strategy="epoch",
            report_to="mlflow",
        )

        mlflow.log_params({
            "model_id": model_hf_id,
            "lora_r": lora_config.r if lora_config else None,
            "epochs": params.get("epochs", 3),
            "learning_rate": params.get("learning_rate", 2e-5),
            "batch_size": params.get("batch_size", 4),
        })

        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, processing_class=tokenizer)
        trainer.train()
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)

        _notify(run_id, "training", logs=f"Modèle sauvegardé dans {output_path}")


def _resolve_dataset_path(run: dict, ds_type: str) -> str:
    """Fetch the dataset file path from the API."""
    ds_id_key = f"{ds_type}_dataset_id"
    ds_id = run.get(ds_id_key) or run["config_snapshot"].get(f"{ds_type}_dataset_id")
    if not ds_id:
        raise ValueError(f"Pas de dataset {ds_type} pour ce run")
    with httpx.Client(base_url=API_URL, timeout=10) as client:
        resp = client.get("/datasets")
        resp.raise_for_status()
        for ds in resp.json():
            if ds["id"] == ds_id:
                return ds["file_path"]
    raise ValueError(f"Dataset {ds_type} id={ds_id} introuvable")


def poll_loop():
    logger.info("Training worker started — polling %s every %ds", API_URL, POLL_INTERVAL)
    while True:
        try:
            with httpx.Client(base_url=API_URL, timeout=30) as client:
                resp = client.get("/runs", params={"status": "pending"})
                resp.raise_for_status()
                runs = resp.json()

            finetune_runs = [r for r in runs if r["task_type"] == "finetune"]
            for run in finetune_runs:
                try:
                    process_run(run)
                    _notify(run["id"], "evaluating", logs="Entraînement terminé, en attente d'évaluation")
                except Exception:
                    tb = traceback.format_exc()
                    logger.error("Run %d failed:\n%s", run["id"], tb)
                    _notify(run["id"], "failed", error=tb)

        except Exception:
            logger.error("Poll error:\n%s", traceback.format_exc())

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    poll_loop()
