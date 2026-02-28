from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class LoraParams(BaseModel):
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class TrainingParamsIn(BaseModel):
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    lora: LoraParams | None = None


class RagasMetricsIn(BaseModel):
    faithfulness: bool = True
    answer_relevancy: bool = True
    context_precision: bool = True
    context_recall: bool = True


class SecurityScanConfigIn(BaseModel):
    modelscan_enabled: bool = True
    training_data_audit: bool = True
    prompt_injection: bool = True
    pii_leakage: bool = True
    toxicity: bool = True
    bias: bool = True
    hallucination: bool = True
    dos_resilience: bool = True
    max_probes_per_category: int = 50
    timeout_per_probe_seconds: int = 300


class RunCreateRequest(BaseModel):
    experiment_name: str = "mlops-default"
    model_name: str
    model_id: str
    task_type: str  # "finetune" | "eval_only" | "security_eval"
    train_dataset_id: int | None = None
    eval_dataset_id: int | None = None
    training_params: TrainingParamsIn | None = None
    ragas_metrics: RagasMetricsIn = Field(default_factory=RagasMetricsIn)
    security_config: SecurityScanConfigIn | None = None
    register_model: bool = False


class DatasetOut(BaseModel):
    id: int
    name: str
    description: str
    file_path: str
    dataset_type: str
    row_count: int
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class RunResultOut(BaseModel):
    metric_name: str
    metric_value: float

    model_config = {"from_attributes": True}


class RunOut(BaseModel):
    id: int
    experiment_name: str
    status: str
    model_name: str
    model_id: str
    task_type: str
    config_snapshot: dict
    mlflow_run_id: str | None
    prefect_flow_run_id: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    finished_at: datetime.datetime | None
    error_message: str | None
    logs: str
    results: list[RunResultOut]

    model_config = {"from_attributes": True}
