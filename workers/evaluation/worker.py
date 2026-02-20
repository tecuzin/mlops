"""Evaluation worker — polls the API for runs awaiting evaluation and processes them."""
from __future__ import annotations

import logging
import os
import time
import traceback

import httpx
import mlflow
from datasets import load_dataset
from ragas import evaluate
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


def _notify(run_id: int, status: str, logs: str = "", metrics: dict | None = None, error: str = ""):
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        client.patch(f"/runs/{run_id}/status", params={"status": status})
        if logs:
            client.patch(f"/runs/{run_id}/logs", params={"text": logs})
        if metrics:
            client.post(f"/runs/{run_id}/results", json=metrics)
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


def process_run(run: dict) -> None:
    run_id = run["id"]
    config = run["config_snapshot"]
    ragas_cfg = run.get("ragas_metrics_config") or config.get("ragas_metrics", {})

    logger.info("Processing evaluation run %d — model %s", run_id, config["model_id"])
    _notify(run_id, "evaluating", logs="Démarrage de l'évaluation RAGAS")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.get("experiment_name", "mlops-default"))

    eval_path = _resolve_eval_path(run)
    ds = load_dataset("json", data_files=eval_path, split="train")

    selected_metrics = [
        metric for name, metric in METRIC_MAP.items()
        if ragas_cfg.get(name, True)
    ]
    metric_names = [m.name for m in selected_metrics]
    _notify(run_id, "evaluating", logs=f"Métriques sélectionnées : {metric_names}")

    result = evaluate(dataset=ds, metrics=selected_metrics)
    scores = {k: v for k, v in result.items() if isinstance(v, (int, float))}

    with mlflow.start_run(run_name=f"{config['model_name']}-eval"):
        mlflow.log_metrics({f"ragas_{k}": v for k, v in scores.items()})

    _notify(run_id, "completed", logs=f"Scores RAGAS : {scores}", metrics=scores)
    logger.info("Run %d completed with scores: %s", run_id, scores)


def poll_loop():
    logger.info("Evaluation worker started — polling %s every %ds", API_URL, POLL_INTERVAL)
    while True:
        try:
            with httpx.Client(base_url=API_URL, timeout=30) as client:
                # Pick up eval_only runs that are pending, and finetune runs that finished training
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
