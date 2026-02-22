"""Training worker — polls the API for pending finetune runs and processes them."""
from __future__ import annotations

import logging
import math
import os
import time
import traceback

import threading

import httpx
import mlflow
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
from transformers import (
    AutoConfig,
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

try:
    mlflow.enable_system_metrics_logging()
    logger.info("MLflow system metrics logging enabled")
except Exception as exc:
    logger.warning("Could not enable system metrics logging: %s", exc)


def _notify(run_id: int, status: str, logs: str = "", metrics: dict | None = None, error: str = ""):
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        if status:
            client.patch(f"/runs/{run_id}/status", params={"status": status})
        if logs:
            client.patch(f"/runs/{run_id}/logs", params={"text": logs})
        if metrics:
            client.post(f"/runs/{run_id}/results", json=metrics)
        if error:
            client.patch(f"/runs/{run_id}/logs", params={"text": f"ERROR: {error}"})


KNOWN_LINEAR_NAMES = {
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "c_attn", "c_proj", "c_fc",
    "query", "key", "value", "dense",
    "qkv_proj", "out_proj",
}


def _find_target_modules(model, requested: list[str]) -> list[str]:
    """Return valid target module names, falling back to auto-detection."""
    all_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    valid = [m for m in requested if m in all_names]
    if valid:
        return valid
    detected = sorted(all_names & KNOWN_LINEAR_NAMES)
    return detected if detected else requested


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


def _log(run_id: int, message: str) -> None:
    """Log locally and push to API."""
    logger.info("[Run %d] %s", run_id, message)
    _notify(run_id, "", logs=message)


def _notify_status(run_id: int, status: str, logs: str = "", **kwargs) -> None:
    """Update status + optional log in one call."""
    logger.info("[Run %d] status -> %s", run_id, status)
    _notify(run_id, status, logs=logs, **kwargs)


def process_run(run: dict) -> None:
    run_id = run["id"]
    config = run["config_snapshot"]
    params = run.get("training_params") or config.get("training_params", {})
    model_hf_id = config["model_id"]
    epochs = params.get("epochs", 3)
    batch_size = params.get("batch_size", 4)
    lr = params.get("learning_rate", 2e-5)

    _notify_status(run_id, "training", logs=(
        f"Démarrage de l'entraînement\n"
        f"  Modèle      : {model_hf_id}\n"
        f"  Epochs       : {epochs}\n"
        f"  Batch size   : {batch_size}\n"
        f"  Learning rate: {lr}"
    ))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.get("experiment_name", "mlops-default"))

    with mlflow.start_run(run_name=config["model_name"]) as mlrun:
        _log(run_id, f"MLflow run ID : {mlrun.info.run_id}")

        # ── Téléchargement des fichiers du modèle ──────────────────
        _log(run_id, f"Téléchargement des fichiers du modèle {model_hf_id}...")
        cache_dir = snapshot_download(
            model_hf_id,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py"],
        )
        _log(run_id, f"Fichiers mis en cache dans {cache_dir}")

        mlflow.set_tag("lifecycle", "new")
        _log(run_id, "Tag MLflow posé : lifecycle=new")

        domain = _detect_domain(run)
        if domain:
            mlflow.set_tag("domain", domain)
            _log(run_id, f"Tag MLflow posé : domain={domain}")

        # ── Chargement du tokenizer ───────────────────────────────
        _log(run_id, f"Chargement du tokenizer {model_hf_id}...")
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _log(run_id, f"Tokenizer chargé (vocab_size={tokenizer.vocab_size})")

        # ── Inspection de la config du modèle ─────────────────────
        model_config = AutoConfig.from_pretrained(cache_dir)
        _log(run_id, (
            f"Architecture du modèle :\n"
            f"  Type                : {model_config.model_type}\n"
            f"  Couches (layers)    : {getattr(model_config, 'num_hidden_layers', '?')}\n"
            f"  Dimension cachée    : {getattr(model_config, 'hidden_size', '?')}\n"
            f"  Têtes d'attention   : {getattr(model_config, 'num_attention_heads', '?')}\n"
            f"  Vocabulaire         : {getattr(model_config, 'vocab_size', '?')}\n"
            f"  Context window      : {getattr(model_config, 'max_position_embeddings', '?')}"
        ))

        # ── Chargement du modèle en mémoire ───────────────────────
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        _log(run_id, f"Chargement du modèle en mémoire ({device_info})...")

        load_done = threading.Event()

        def _progress_reporter():
            """Report progress while model loads."""
            elapsed = 0
            while not load_done.wait(timeout=15):
                elapsed += 15
                _log(run_id, f"  Chargement en cours... ({elapsed}s écoulées)")

        reporter = threading.Thread(target=_progress_reporter, daemon=True)
        reporter.start()

        model = AutoModelForCausalLM.from_pretrained(cache_dir)
        load_done.set()
        reporter.join(timeout=2)

        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        _log(run_id, (
            f"Modèle chargé !\n"
            f"  Paramètres totaux   : {total_params / 1e6:.1f}M\n"
            f"  Taille en mémoire   : {model_size_mb:.0f} MB\n"
            f"  Dtype               : {next(model.parameters()).dtype}"
        ))

        lora_config = _build_lora_config(params)
        if lora_config:
            requested = list(lora_config.target_modules)
            resolved = _find_target_modules(model, requested)
            if resolved != requested:
                _log(run_id, (
                    f"Modules LoRA demandés ({requested}) absents du modèle.\n"
                    f"  Modules détectés automatiquement : {resolved}"
                ))
                lora_config.target_modules = resolved
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _log(run_id, (
                f"LoRA appliqué (r={lora_config.r}, alpha={lora_config.lora_alpha})\n"
                f"  Modules ciblés          : {resolved}\n"
                f"  Paramètres entraînables : {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M "
                f"({100 * trainable / total_params:.2f}%)"
            ))

        train_ds_path = _resolve_dataset_path(run, "train")
        _log(run_id, f"Chargement du dataset : {train_ds_path}")
        ds = load_dataset("json", data_files=train_ds_path, split="train")
        _log(run_id, f"Dataset chargé — {len(ds)} exemples")

        max_len = params.get("max_seq_length", 512)

        def tokenize(example):
            prompt = f"### Question: {example['question']}\n### Answer: {example['answer']}"
            encoded = tokenizer(prompt, truncation=True, max_length=max_len, padding="max_length")
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        _log(run_id, f"Tokenisation du dataset (max_seq_length={max_len})...")
        train_ds = ds.map(tokenize, batched=False, remove_columns=ds.column_names)
        _log(run_id, "Tokenisation terminée")

        output_path = f"{OUTPUT_DIR}/{config['model_name']}"
        grad_accum = params.get("gradient_accumulation_steps", 4)
        fp16 = params.get("fp16", True)
        warmup = params.get("warmup_steps", 100)

        _log(run_id, (
            f"Configuration de l'entraînement :\n"
            f"  Output dir               : {output_path}\n"
            f"  Gradient accumulation     : {grad_accum}\n"
            f"  FP16                      : {fp16}\n"
            f"  Warmup steps              : {warmup}\n"
            f"  Logging steps             : 10\n"
            f"  Save strategy             : epoch"
        ))

        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            warmup_steps=warmup,
            gradient_accumulation_steps=grad_accum,
            fp16=fp16,
            logging_steps=10,
            save_strategy="epoch",
            report_to="mlflow",
        )

        mlflow.log_params({
            "model_id": model_hf_id,
            "lora_r": lora_config.r if lora_config else None,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
        })

        mlflow.set_tag("lifecycle", "training")
        _log(run_id, "Tag MLflow posé : lifecycle=training")

        _log(run_id, "Lancement de l'entraînement...")
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, processing_class=tokenizer)
        train_result = trainer.train()

        train_loss = train_result.metrics.get("train_loss", 0)
        if not isinstance(train_loss, (int, float)):
            train_loss = 0.0
        train_runtime = train_result.metrics.get("train_runtime", 0)
        train_samples_per_sec = train_result.metrics.get("train_samples_per_second", 0)
        train_steps_per_sec = train_result.metrics.get("train_steps_per_second", 0)
        total_flos = train_result.metrics.get("total_flos", 0)
        perplexity = math.exp(train_loss) if 0 < train_loss < 100 else 0.0

        _log(run_id, (
            f"Entraînement terminé !\n"
            f"  Loss finale              : {train_loss:.6f}\n"
            f"  Perplexité               : {perplexity:.2f}\n"
            f"  Durée                    : {train_runtime:.1f}s\n"
            f"  Samples/sec              : {train_samples_per_sec:.2f}\n"
            f"  Steps/sec                : {train_steps_per_sec:.2f}\n"
            f"  Steps totaux             : {train_result.global_step}\n"
            f"  Total FLOPs              : {total_flos:.2e}"
        ))

        # ── Merge LoRA weights for clean inference ────────────────
        if lora_config:
            _log(run_id, "Fusion des poids LoRA dans le modèle de base...")
            model = model.merge_and_unload()
            _log(run_id, "Fusion LoRA terminée — modèle complet prêt pour l'inférence")

        _log(run_id, f"Sauvegarde du modèle fusionné dans {output_path}...")
        model.config._name_or_path = model_hf_id
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        _log(run_id, "Modèle et tokenizer sauvegardés")

        mlflow.set_tag("lifecycle", "finetuned")
        _log(run_id, "Tag MLflow posé : lifecycle=finetuned")

        # ── Log comprehensive metrics to MLflow ───────────────────
        extra_metrics = {
            "final_train_loss": float(train_loss),
            "perplexity": perplexity,
            "train_runtime_seconds": float(train_runtime),
            "train_samples_per_second": float(train_samples_per_sec),
            "train_steps_per_second": float(train_steps_per_sec),
            "total_flos": float(total_flos),
            "total_steps": float(train_result.global_step),
            "trainable_params": float(sum(p.numel() for p in model.parameters())),
            "model_size_mb": float(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)),
        }
        mlflow.log_metrics(extra_metrics)
        _log(run_id, "Métriques supplémentaires enregistrées dans MLflow")

        # ── Register model in MLflow Model Registry ───────────────
        _log(run_id, "Enregistrement des artefacts dans MLflow...")
        mlflow.log_artifacts(output_path, "model")
        model_uri = f"runs:/{mlrun.info.run_id}/model"
        registered_name = config["model_name"]
        _log(run_id, f"Enregistrement dans le Model Registry : {registered_name}")
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            try:
                client.create_registered_model(registered_name)
                _log(run_id, f"Registered model '{registered_name}' créé")
            except Exception:
                _log(run_id, f"Registered model '{registered_name}' existe déjà")
            mv = client.create_model_version(
                name=registered_name,
                source=model_uri,
                run_id=mlrun.info.run_id,
            )
            _log(run_id, f"Modèle enregistré : {registered_name} v{mv.version}")

            client.set_model_version_tag(registered_name, mv.version, "lifecycle", "finetuned")
            if domain:
                client.set_model_version_tag(registered_name, mv.version, "domain", domain)
            _log(run_id, f"Tags lifecycle/domain posés sur la Model Version {mv.version}")

            mlflow.set_tag("lifecycle", "registered")
            _log(run_id, "Tag MLflow posé : lifecycle=registered (modèle publié)")

            with httpx.Client(base_url=API_URL, timeout=10) as http:
                http.patch(f"/runs/{run_id}/mlflow-model", json={
                    "mlflow_run_id": mlrun.info.run_id,
                    "mlflow_model_name": registered_name,
                    "mlflow_model_version": str(mv.version),
                })
            _log(run_id, "Infos MLflow model persistées dans l'API")
        except Exception as reg_err:
            _log(run_id, f"Avertissement Model Registry : {reg_err}")

        # ── Send training metrics to API ──────────────────────────
        api_metrics = {
            "train_loss": float(train_loss),
            "perplexity": perplexity,
            "train_runtime": float(train_runtime),
            "train_samples_per_second": float(train_samples_per_sec),
        }
        _notify(run_id, "", metrics=api_metrics)


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


def _detect_domain(run: dict) -> str | None:
    """Detect domain (medic/legal) from the training dataset name."""
    ds_id = run.get("train_dataset_id") or run["config_snapshot"].get("train_dataset_id")
    if not ds_id:
        return None
    try:
        with httpx.Client(base_url=API_URL, timeout=10) as client:
            resp = client.get("/datasets")
            resp.raise_for_status()
            for ds in resp.json():
                if ds["id"] == ds_id:
                    name = ds["name"].lower()
                    if "medical" in name:
                        return "medic"
                    if "legal" in name:
                        return "legal"
                    return None
    except Exception as exc:
        logger.warning("Could not detect domain: %s", exc)
    return None


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
