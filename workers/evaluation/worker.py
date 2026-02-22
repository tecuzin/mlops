"""Evaluation worker — loads trained models, generates answers, runs RAGAS evaluation."""
from __future__ import annotations

import logging
import math
import os
import threading
import time
import traceback

import httpx
import mlflow
import torch
from datasets import Dataset as HFDataset, load_dataset
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
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://api:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")
MLSCORE_THRESHOLD = float(os.getenv("MLSCORE_THRESHOLD", "0.7"))

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

MLSCORE_WEIGHTS = {
    "faithfulness": 0.30,
    "answer_relevancy": 0.20,
    "context_precision": 0.25,
    "context_recall": 0.25,
}

try:
    mlflow.enable_system_metrics_logging()
    logger.info("MLflow system metrics logging enabled")
except Exception as exc:
    logger.warning("Could not enable system metrics logging: %s", exc)


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


def _detect_domain(run: dict) -> str | None:
    """Detect domain (medic/legal) from the eval or train dataset name."""
    for key in ("eval_dataset_id", "train_dataset_id"):
        ds_id = run.get(key) or run["config_snapshot"].get(key)
        if not ds_id:
            continue
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
        except Exception as exc:
            logger.warning("Could not detect domain: %s", exc)
    return None


def _log(run_id: int, message: str) -> None:
    logger.info("[Run %d] %s", run_id, message)
    _notify(run_id, "", logs=message)


def _notify_status(run_id: int, status: str, logs: str = "", **kwargs) -> None:
    logger.info("[Run %d] status -> %s", run_id, status)
    _notify(run_id, status, logs=logs, **kwargs)


# ── Model Inference ──────────────────────────────────────────────────


def _load_trained_model(model_output_path: str, run_id: int):
    """Load the trained (merged) model and tokenizer from disk."""
    _log(run_id, f"Chargement du modèle entraîné depuis {model_output_path}...")

    load_done = threading.Event()

    def _progress_reporter():
        elapsed = 0
        while not load_done.wait(timeout=10):
            elapsed += 10
            _log(run_id, f"  Chargement du modèle en cours... ({elapsed}s)")

    reporter = threading.Thread(target=_progress_reporter, daemon=True)
    reporter.start()

    model = AutoModelForCausalLM.from_pretrained(model_output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_output_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_done.set()
    reporter.join(timeout=2)

    total_params = sum(p.numel() for p in model.parameters())
    _log(run_id, (
        f"Modèle entraîné chargé\n"
        f"  Paramètres : {total_params / 1e6:.1f}M\n"
        f"  Dtype      : {next(model.parameters()).dtype}"
    ))
    return model, tokenizer


def _generate_answers(model, tokenizer, ds, run_id: int) -> list[str]:
    """Generate answers using the trained model for each evaluation sample."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    max_pos = getattr(model.config, "max_position_embeddings", 512)
    _log(run_id, f"Context window du modèle : {max_pos} tokens")

    generated_answers = []
    n = len(ds)
    _log(run_id, f"Début de l'inférence — {n} samples sur {device.upper()}")

    for i, row in enumerate(ds):
        question = row["question"]
        contexts = row.get("contexts", [])
        ctx_text = "\n".join(contexts) if contexts else ""

        prompt = f"Context: {ctx_text}\n\nQuestion: {question}\n\nAnswer:"

        reserve_for_generation = min(50, max_pos // 4)
        max_input_len = max_pos - reserve_for_generation

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_len
        ).to(device)

        input_len = inputs["input_ids"].shape[1]
        max_new = max_pos - input_len

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as gen_err:
            _log(run_id, f"  [{i+1}/{n}] Erreur de génération: {gen_err} — fallback vide")
            generated_text = "(generation failed)"

        generated_answers.append(generated_text)
        preview = generated_text[:120].replace("\n", " ")
        _log(run_id, f"  [{i+1}/{n}] Réponse générée ({len(generated_text)} chars): {preview}...")

    _log(run_id, f"Inférence terminée — {n} réponses générées")
    return generated_answers


# ── MLScore ──────────────────────────────────────────────────────────


def _compute_mlscore(scores: dict[str, float]) -> float:
    total_w = 0.0
    weighted = 0.0
    for metric, weight in MLSCORE_WEIGHTS.items():
        val = scores.get(metric, 0.0)
        weighted += _safe_float(val) * weight
        total_w += weight
    return round(weighted / total_w, 4) if total_w > 0 else 0.0


# ── Main processing ─────────────────────────────────────────────────


def process_run(run: dict) -> None:
    run_id = run["id"]
    config = run["config_snapshot"]
    ragas_cfg = run.get("ragas_metrics_config") or config.get("ragas_metrics", {})
    model_name = config.get("model_name", "unknown")
    task_type = run.get("task_type", config.get("task_type", ""))

    _notify_status(run_id, "evaluating", logs=(
        f"Démarrage de l'évaluation\n"
        f"  Modèle : {config['model_id']}\n"
        f"  Run    : {model_name}\n"
        f"  Task   : {task_type}"
    ))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.get("experiment_name", "mlops-default"))

    domain = _detect_domain(run)

    with mlflow.start_run(run_name=f"{model_name}-eval") as mlrun:
        # ── Phase: evaluating started ─────────────────────────────
        mlflow.set_tag("lifecycle", "evaluating")
        if domain:
            mlflow.set_tag("domain", domain)
            _log(run_id, f"Tags MLflow initiaux : lifecycle=evaluating, domain={domain}")
        else:
            _log(run_id, "Tag MLflow initial : lifecycle=evaluating")

        # ── Load evaluation dataset ───────────────────────────────
        eval_path = _resolve_eval_path(run)
        _log(run_id, f"Chargement du dataset d'évaluation : {eval_path}")
        ds = load_dataset("json", data_files=eval_path, split="train")
        _log(run_id, f"Dataset chargé — {len(ds)} exemples, colonnes: {ds.column_names}")

        # ── Model inference phase (if trained model available) ────
        model_output_path = os.path.join(OUTPUT_DIR, model_name)
        use_model_inference = task_type == "finetune" and os.path.isdir(model_output_path)

        if use_model_inference:
            mlflow.set_tag("lifecycle", "inference")
            _log(run_id, "=== PHASE D'INFÉRENCE — Génération des réponses par le modèle entraîné ===")
            model, tokenizer = _load_trained_model(model_output_path, run_id)
            generated_answers = _generate_answers(model, tokenizer, ds, run_id)

            original_answers = ds["answer"]
            ds = ds.map(
                lambda example, idx: {"answer": generated_answers[idx]},
                with_indices=True,
            )
            _log(run_id, "Réponses du dataset remplacées par les réponses du modèle")

            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            _log(run_id, "Mémoire du modèle libérée")
        else:
            original_answers = None
            _log(run_id, "Pas de modèle entraîné disponible — utilisation des réponses pré-écrites du dataset")

        # ── Select RAGAS metrics ──────────────────────────────────
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
            f"  {n_samples} exemples × {n_metrics} métriques = ~{total_calls} appels LLM estimés"
        ))

        # ── Initialize LLM judge ──────────────────────────────────
        mistral_model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        _log(run_id, f"Initialisation du LLM juge : Mistral ({mistral_model})")
        evaluator_llm = LangchainLLMWrapper(ChatMistralAI(model=mistral_model, temperature=0))
        _log(run_id, "Initialisation des embeddings Mistral")
        evaluator_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings())
        _log(run_id, "LLM et embeddings prêts")

        # ── Run RAGAS evaluation ──────────────────────────────────
        mlflow.set_tag("lifecycle", "ragas_running")
        _log(run_id, "=== PHASE D'ÉVALUATION RAGAS ===")

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

        # ── Process per-sample scores ─────────────────────────────
        per_sample = result.scores
        _log(run_id, f"Scores par sample ({len(per_sample)} exemples) :")
        for i, row_scores in enumerate(per_sample):
            parts = []
            for k, v in row_scores.items():
                if isinstance(v, (int, float)):
                    parts.append(f"{k}={v:.4f}" if math.isfinite(v) else f"{k}=NaN")
            answer_preview = (ds[i]["answer"] or "")[:80].replace("\n", " ")
            _log(run_id, f"  Sample {i+1}/{len(per_sample)}: {', '.join(parts)}")
            if use_model_inference:
                _log(run_id, f"    Réponse modèle : {answer_preview}...")
                if original_answers:
                    ref_preview = (original_answers[i] or "")[:80].replace("\n", " ")
                    _log(run_id, f"    Réponse réf.   : {ref_preview}...")

        # ── Compute average scores ────────────────────────────────
        scores: dict[str, float] = {}
        if per_sample:
            metric_keys = [k for k in per_sample[0] if isinstance(per_sample[0][k], (int, float))]
            for key in metric_keys:
                vals = [_safe_float(s[key]) for s in per_sample if isinstance(s.get(key), (int, float))]
                scores[key] = sum(vals) / len(vals) if vals else 0.0

        # ── Compute MLScore ───────────────────────────────────────
        mlscore = _compute_mlscore(scores)
        scores["ml_score"] = mlscore

        scores_display = "\n".join(f"  {k:<25}: {v:.4f}" for k, v in scores.items())
        _log(run_id, f"=== RÉSULTATS FINAUX ===\n{scores_display}")
        _log(run_id, f"MLScore composite         : {mlscore:.4f}")
        if use_model_inference:
            _log(run_id, "(Scores basés sur les réponses générées par le modèle entraîné)")
        else:
            _log(run_id, "(Scores basés sur les réponses pré-écrites du dataset)")

        # ── Validation tag ────────────────────────────────────────
        validation_tag = "validated" if mlscore >= MLSCORE_THRESHOLD else "rejected"
        _log(run_id, f"Validation : {validation_tag} (MLScore={mlscore:.4f}, seuil={MLSCORE_THRESHOLD})")

        # ── Log metrics & final tags to MLflow ────────────────────
        safe_mlflow = {f"ragas_{k}": _safe_float(v) for k, v in scores.items()}
        safe_mlflow["ml_score"] = mlscore
        safe_mlflow["inference_mode"] = 1.0 if use_model_inference else 0.0
        safe_mlflow["eval_samples"] = float(n_samples)
        mlflow.log_metrics(safe_mlflow)

        mlflow.log_params({
            "model_name": model_name,
            "model_id": config["model_id"],
            "eval_dataset": eval_path,
            "task_type": task_type,
            "inference_mode": "model" if use_model_inference else "dataset",
            "llm_judge": mistral_model,
            "ragas_metrics": str(metric_names),
        })

        mlflow.set_tag("lifecycle", "completed")
        mlflow.set_tag("validation", validation_tag)
        mlflow.set_tag("ml_score_threshold", str(MLSCORE_THRESHOLD))
        _log(run_id, f"Tags MLflow finaux : lifecycle=completed, validation={validation_tag}")

        if use_model_inference and per_sample:
            inference_table = []
            for i, row_scores in enumerate(per_sample):
                entry = {
                    "question": ds[i]["question"],
                    "model_answer": ds[i]["answer"],
                    "ground_truth": ds[i].get("ground_truth", ""),
                    **{k: _safe_float(v) for k, v in row_scores.items() if isinstance(v, (int, float))},
                }
                inference_table.append(entry)
            try:
                import json
                table_path = "/tmp/inference_results.json"
                with open(table_path, "w") as f:
                    json.dump(inference_table, f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(table_path, "evaluation")
                _log(run_id, "Table d'inférence sauvegardée dans MLflow")
            except Exception as e:
                _log(run_id, f"Avertissement sauvegarde table : {e}")

        _log(run_id, f"Métriques enregistrées dans MLflow (run ID : {mlrun.info.run_id})")

    # ── Persist MLflow run ID in API ──────────────────────────────
    try:
        with httpx.Client(base_url=API_URL, timeout=10) as http:
            http.patch(f"/runs/{run_id}/mlflow-model", json={
                "mlflow_run_id": mlrun.info.run_id,
            })
        _log(run_id, "MLflow run ID persisté dans l'API")
    except Exception as e:
        _log(run_id, f"Avertissement persistance mlflow_run_id : {e}")

    # ── Set tags on Model Version (finetune only) ─────────────────
    if task_type == "finetune":
        try:
            from mlflow.tracking import MlflowClient
            with httpx.Client(base_url=API_URL, timeout=10) as http:
                run_resp = http.get(f"/runs/{run_id}")
                run_resp.raise_for_status()
                run_data = run_resp.json()
            mv_name = run_data.get("mlflow_model_name")
            mv_version = run_data.get("mlflow_model_version")
            if mv_name and mv_version:
                client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
                client.set_model_version_tag(mv_name, mv_version, "validation", validation_tag)
                _log(run_id, f"Tag validation={validation_tag} posé sur Model Version {mv_name} v{mv_version}")
            else:
                _log(run_id, "Pas d'info Model Version — tags Model Registry ignorés")
        except Exception as mv_err:
            _log(run_id, f"Avertissement tags Model Version : {mv_err}")

    _notify_status(run_id, "completed", logs="Évaluation terminée avec succès", metrics=scores)
    logger.info("Run %d completed — MLScore=%.4f, scores: %s", run_id, mlscore, scores)


TAG_SYNC_INTERVAL = int(os.getenv("TAG_SYNC_INTERVAL", "3600"))


def _reconcile_run_tags(run_data: dict, datasets: dict[int, str]) -> None:
    """Re-apply all expected MLflow tags for a single pipeline run."""
    from mlflow.tracking import MlflowClient

    run_id = run_data["id"]
    config = run_data.get("config_snapshot", {})
    status = run_data["status"]
    task_type = run_data.get("task_type", "")

    lifecycle = None
    if status == "completed" and task_type == "finetune":
        lifecycle = "finetuned"
    elif status == "completed" and task_type == "eval_only":
        lifecycle = "completed"
    elif status == "training":
        lifecycle = "training"
    elif status == "evaluating":
        lifecycle = "evaluating"

    domain = None
    for key in ("train_dataset_id", "eval_dataset_id"):
        ds_id = run_data.get(key) or config.get(key)
        if ds_id and ds_id in datasets:
            name = datasets[ds_id].lower()
            if "medical" in name:
                domain = "medic"
                break
            if "legal" in name:
                domain = "legal"
                break

    validation = None
    for r in run_data.get("results", []):
        if r["metric_name"] == "ml_score":
            validation = "validated" if r["metric_value"] >= MLSCORE_THRESHOLD else "rejected"
            break

    mv_name = run_data.get("mlflow_model_name")
    mv_version = run_data.get("mlflow_model_version")
    if mv_name and mv_version:
        try:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            if lifecycle:
                client.set_model_version_tag(mv_name, mv_version, "lifecycle", lifecycle)
            if domain:
                client.set_model_version_tag(mv_name, mv_version, "domain", domain)
            if validation:
                client.set_model_version_tag(mv_name, mv_version, "validation", validation)
            logger.info("[TagSync] Run %d — Model Version %s v%s tags updated", run_id, mv_name, mv_version)
        except Exception as e:
            logger.warning("[TagSync] Run %d — Model Version tag error: %s", run_id, e)

    mlflow_run_id = run_data.get("mlflow_run_id")
    if mlflow_run_id:
        try:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            if lifecycle:
                client.set_tag(mlflow_run_id, "lifecycle", lifecycle)
            if domain:
                client.set_tag(mlflow_run_id, "domain", domain)
            if validation:
                client.set_tag(mlflow_run_id, "validation", validation)
                client.set_tag(mlflow_run_id, "ml_score_threshold", str(MLSCORE_THRESHOLD))
            logger.info("[TagSync] Run %d — MLflow run %s tags updated", run_id, mlflow_run_id)
        except Exception as e:
            logger.warning("[TagSync] Run %d — MLflow run tag error: %s", run_id, e)


def _tag_sync_loop():
    """Background thread: reconcile all MLflow tags every TAG_SYNC_INTERVAL seconds."""
    logger.info("[TagSync] Started — reconciliation every %ds", TAG_SYNC_INTERVAL)
    while True:
        time.sleep(TAG_SYNC_INTERVAL)
        try:
            logger.info("[TagSync] Running tag reconciliation...")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            with httpx.Client(base_url=API_URL, timeout=30) as client:
                ds_resp = client.get("/datasets")
                ds_resp.raise_for_status()
                datasets = {d["id"]: d["name"] for d in ds_resp.json()}

                runs_resp = client.get("/runs")
                runs_resp.raise_for_status()
                all_runs = runs_resp.json()

            synced = 0
            for run in all_runs:
                if run["status"] in ("completed", "failed", "training", "evaluating"):
                    _reconcile_run_tags(run, datasets)
                    synced += 1

            logger.info("[TagSync] Reconciliation complete — %d runs synced", synced)
        except Exception:
            logger.error("[TagSync] Error:\n%s", traceback.format_exc())


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
    sync_thread = threading.Thread(target=_tag_sync_loop, daemon=True)
    sync_thread.start()
    logger.info("Tag sync thread launched (interval: %ds)", TAG_SYNC_INTERVAL)
    poll_loop()
