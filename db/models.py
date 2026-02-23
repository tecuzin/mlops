from __future__ import annotations

import datetime
import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SECURITY_SCANNING = "security_scanning"
    COMPLETED = "completed"
    FAILED = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, default="")
    file_path = Column(String(500), nullable=False)
    dataset_type = Column(String(50), nullable=False)  # "train" | "eval"
    row_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(255), nullable=False)
    status = Column(Enum(RunStatus), default=RunStatus.PENDING)
    config_snapshot = Column(JSON, nullable=False)
    mlflow_run_id = Column(String(255), nullable=True)
    prefect_flow_run_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    finished_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    logs = Column(Text, default="")

    model_name = Column(String(255), nullable=False)
    model_id = Column(String(255), nullable=False)
    task_type = Column(String(50), nullable=False)
    train_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    eval_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)

    training_params = Column(JSON, nullable=True)
    ragas_metrics_config = Column(JSON, nullable=True)
    security_config = Column(JSON, nullable=True)
    register_model = Column(Boolean, default=False)

    results = relationship("RunResult", back_populates="run", cascade="all, delete-orphan")
    train_dataset = relationship("Dataset", foreign_keys=[train_dataset_id])
    eval_dataset = relationship("Dataset", foreign_keys=[eval_dataset_id])


class RunResult(Base):
    __tablename__ = "run_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    run = relationship("PipelineRun", back_populates="results")
