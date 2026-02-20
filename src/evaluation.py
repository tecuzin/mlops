from __future__ import annotations

import logging

import mlflow
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import ModelConfig, RagasMetrics

logger = logging.getLogger(__name__)

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def _select_metrics(ragas_cfg: RagasMetrics) -> list:
    selected = []
    for name, metric in METRIC_MAP.items():
        if getattr(ragas_cfg, name, False):
            selected.append(metric)
    return selected


def evaluate_model(model_cfg: ModelConfig) -> dict[str, float]:
    """Run RAGAS evaluation and return metric scores."""
    logger.info("Loading eval dataset from %s", model_cfg.dataset.eval_path)
    ds = load_dataset("json", data_files=model_cfg.dataset.eval_path, split="train")

    metrics = _select_metrics(model_cfg.ragas_metrics)
    metric_names = [m.name for m in metrics]
    logger.info("Running RAGAS evaluation with metrics: %s", metric_names)

    result = evaluate(dataset=ds, metrics=metrics)
    scores = {k: v for k, v in result.items() if isinstance(v, (int, float))}

    mlflow.log_metrics({f"ragas_{k}": v for k, v in scores.items()})
    logger.info("RAGAS scores for %s: %s", model_cfg.name, scores)

    return scores
