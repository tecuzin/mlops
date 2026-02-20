"""Evaluation worker — polls the API for runs awaiting evaluation and processes them."""
from __future__ import annotations

import logging
import math
import os
import threading
import time
import traceback

import httpx
import mlflow
from datasets import load_dataset
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://api:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def _safe_float(v) -> float:
    """Convert to a JSON-safe float (NaN/Inf become 0.0)."""
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return 0.0


def _notify(run_id: int, status: str, logs: str = "", metrics: dict | None = None, error: str = ""):
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        if status:
            client.patch(f"/runs/{run_id}/status", params={"status": status})
        if logs:
            client.patch(f"/runs/{run_id}/logs", params={"text": logs})
        if metrics:
            safe = {k: _safe_float(v) for k, v in metrics.items()}
            client.post(f"/runs/{run_id}/results", json=safe)
        if error:
            client.patch(f"/runs/{run_id}/logs", params={"text": f"ERROR: {error}"})


def _resolve_eval_path(run: dict) -> str:
    ds_id = run.get("eval_dataset_id") or run["config_snapshot"].get("eval_dataset_id")
    if not ds_id:
        raise ValueError("Pas de dataset d'évaluation pour ce run")
    with httpx.Client(base_url=API_URL, timeout=10) as client:
        resp = client.get("/datasets")
        resp.raise_for_status()
        for ds in resp.json():
            if ds["id"] == ds_id:
                return ds["file_path"]
    raise ValueError(f"Dataset eval id={ds_id} introuvable")


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
    ragas_cfg = run.get("ragas_metrics_config") or config.get("ragas_metrics", {})
    model_name = config.get("model_name", "unknown")

    _notify_status(run_id, "evaluating", logs=(
        f"Démarrage de l'évaluation RAGAS\n"
        f"  Modèle : {config['model_id']}\n"
        f"  Run    : {model_name}"
    ))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.get("experiment_name", "mlops-default"))

    eval_path = _resolve_eval_path(run)
    _log(run_id, f"Chargement du dataset d'évaluation : {eval_path}")
    ds = load_dataset("json", data_files=eval_path, split="train")
    _log(run_id, f"Dataset chargé — {len(ds)} exemples")

    columns = list(ds.column_names)
    _log(run_id, f"Colonnes du dataset : {columns}")

    selected_metrics = [
        metric for name, metric in METRIC_MAP.items()
        if ragas_cfg.get(name, True)
    ]
    metric_names = [m.name for m in selected_metrics]
    n_samples = len(ds)
    n_metrics = len(selected_metrics)
    total_calls = n_samples * n_metrics
    _log(run_id, (
        f"Métriques RAGAS sélectionnées : {metric_names}\n"
        f"  {n_samples} exemples × {n_metrics} métriques = {total_calls} appels LLM estimés"
    ))

    mistral_model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    _log(run_id, f"Initialisation du LLM juge : Mistral ({mistral_model})")
    evaluator_llm = LangchainLLMWrapper(ChatMistralAI(model=mistral_model, temperature=0))
    _log(run_id, "Initialisation des embeddings Mistral")
    evaluator_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings())
    _log(run_id, "LLM et embeddings prêts")

    _log(run_id, "Lancement de l'évaluation RAGAS (cela peut prendre plusieurs minutes)...")

    eval_done = threading.Event()

    def _progress_reporter():
        elapsed = 0
        while not eval_done.wait(timeout=20):
            elapsed += 20
            _log(run_id, f"  Évaluation en cours... ({elapsed}s écoulées)")

    reporter = threading.Thread(target=_progress_reporter, daemon=True)
    reporter.start()

    result = evaluate(
        dataset=ds,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    eval_done.set()
    reporter.join(timeout=3)

    _log(run_id, "Évaluation RAGAS terminée — traitement des résultats")

    per_sample = result.scores
    _log(run_id, f"Scores par sample ({len(per_sample)} exemples) :")
    for i, row_scores in enumerate(per_sample):
        parts = []
        for k, v in row_scores.items():
            if isinstance(v, (int, float)):
                if math.isfinite(v):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}=NaN")
        _log(run_id, f"  Sample {i+1}/{len(per_sample)}: {', '.join(parts)}")

    scores: dict[str, float] = {}
    if per_sample:
        metric_keys = [k for k in per_sample[0] if isinstance(per_sample[0][k], (int, float))]
        for key in metric_keys:
            vals = [_safe_float(s[key]) for s in per_sample if isinstance(s.get(key), (int, float))]
            scores[key] = sum(vals) / len(vals) if vals else 0.0

    scores_display = "\n".join(f"  {k:<25}: {v:.4f}" for k, v in scores.items())
    _log(run_id, f"Scores moyens :\n{scores_display}")

    with mlflow.start_run(run_name=f"{model_name}-eval") as mlrun:
        safe_mlflow = {f"ragas_{k}": _safe_float(v) for k, v in scores.items()}
        mlflow.log_metrics(safe_mlflow)
        _log(run_id, f"Métriques enregistrées dans MLflow (run ID : {mlrun.info.run_id})")

    _notify_status(run_id, "completed", logs="Évaluation terminée avec succès", metrics=scores)
    logger.info("Run %d completed with scores: %s", run_id, scores)


def poll_loop():
    logger.info("Evaluation worker started — polling %s every %ds", API_URL, POLL_INTERVAL)
    while True:
        try:
            with httpx.Client(base_url=API_URL, timeout=30) as client:
                pending_resp = client.get("/runs", params={"status": "pending"})
                pending_resp.raise_for_status()
                eval_only = [r for r in pending_resp.json() if r["task_type"] == "eval_only"]

                evaluating_resp = client.get("/runs", params={"status": "evaluating"})
                evaluating_resp.raise_for_status()
                awaiting_eval = evaluating_resp.json()

            for run in eval_only + awaiting_eval:
                try:
                    process_run(run)
                except Exception:
                    tb = traceback.format_exc()
                    logger.error("Run %d failed:\n%s", run["id"], tb)
                    _notify(run["id"], "failed", error=tb)

        except Exception:
            logger.error("Poll error:\n%s", traceback.format_exc())

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    poll_loop()
