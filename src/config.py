from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    FINETUNE = "finetune"
    EVAL_ONLY = "eval_only"


class LoraConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class TrainingParams(BaseModel):
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    lora: LoraConfig | None = None


class DatasetConfig(BaseModel):
    train_path: str | None = None
    eval_path: str


class RagasMetrics(BaseModel):
    faithfulness: bool = True
    answer_relevancy: bool = True
    context_precision: bool = True
    context_recall: bool = True


class ModelConfig(BaseModel):
    name: str
    model_id: str
    task: TaskType
    dataset: DatasetConfig
    training_params: TrainingParams | None = None
    ragas_metrics: RagasMetrics = Field(default_factory=RagasMetrics)
    register_model: bool = False


class PipelineConfig(BaseModel):
    experiment_name: str = "mlops-default"
    models: list[ModelConfig]


def load_config(path: str | Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
