from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from prefect import flow, task

from src.config import ModelConfig, PipelineConfig, TaskType, load_config
from src.evaluation import evaluate_model
from src.training import train_model

logger = logging.getLogger(__name__)


@task(name="finetune", retries=1, retry_delay_seconds=30)
def finetune_task(model_cfg: ModelConfig, output_dir: str) -> Path:
    return train_model(model_cfg, output_dir=output_dir)


@task(name="evaluate_ragas", retries=1, retry_delay_seconds=10)
def evaluate_task(model_cfg: ModelConfig) -> dict[str, float]:
    return evaluate_model(model_cfg)


@task(name="register_model")
def register_task(model_cfg: ModelConfig, model_path: Path) -> None:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=None,
        artifacts={"model_dir": str(model_path)},
        registered_model_name=model_cfg.name,
    )
    logger.info("Registered model %s in MLflow Model Registry", model_cfg.name)


@flow(name="mlops_pipeline", log_prints=True)
def mlops_pipeline(config_path: str, output_dir: str = "outputs") -> None:
    config = load_config(config_path)
    mlflow.set_experiment(config.experiment_name)

    for model_cfg in config.models:
        with mlflow.start_run(run_name=model_cfg.name):
            mlflow.set_tags({
                "model_id": model_cfg.model_id,
                "task": model_cfg.task.value,
                "dataset_eval": model_cfg.dataset.eval_path,
            })

            model_path: Path | None = None

            if model_cfg.task == TaskType.FINETUNE:
                model_path = finetune_task(model_cfg, output_dir)

            scores = evaluate_task(model_cfg)
            print(f"[{model_cfg.name}] RAGAS scores: {scores}")

            if model_cfg.register_model and model_path is not None:
                register_task(model_cfg, model_path)
