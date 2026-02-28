from __future__ import annotations

import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from db.models import Base, Dataset, PipelineRun, RunResult, RunStatus
from db.session import engine, get_db

from .schemas import DatasetOut, RunCreateRequest, RunOut

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="MLOps API", version="1.0.0", lifespan=lifespan)


# ── Datasets ──────────────────────────────────────────────────────────

@app.get("/datasets", response_model=list[DatasetOut])
def list_datasets(dataset_type: str | None = None, db: Session = Depends(get_db)):
    q = db.query(Dataset)
    if dataset_type:
        q = q.filter(Dataset.dataset_type == dataset_type)
    return q.order_by(Dataset.name).all()


# ── Runs ──────────────────────────────────────────────────────────────

@app.post("/runs", response_model=RunOut, status_code=201)
def create_run(req: RunCreateRequest, db: Session = Depends(get_db)):
    if req.task_type == "security_eval":
        if req.security_config is None:
            raise HTTPException(400, "security_config est requis pour une évaluation de sécurité")
    else:
        if req.eval_dataset_id is None:
            raise HTTPException(400, "eval_dataset_id est requis pour ce type de tâche")
        eval_ds = db.query(Dataset).get(req.eval_dataset_id)
        if not eval_ds:
            raise HTTPException(404, "Dataset d'évaluation introuvable")

    if req.task_type == "finetune" and req.train_dataset_id is None:
        raise HTTPException(400, "Un dataset d'entraînement est requis pour le fine-tuning")

    if req.train_dataset_id:
        train_ds = db.query(Dataset).get(req.train_dataset_id)
        if not train_ds:
            raise HTTPException(404, "Dataset d'entraînement introuvable")

    run = PipelineRun(
        experiment_name=req.experiment_name,
        status=RunStatus.PENDING,
        model_name=req.model_name,
        model_id=req.model_id,
        task_type=req.task_type,
        train_dataset_id=req.train_dataset_id,
        eval_dataset_id=req.eval_dataset_id,
        training_params=req.training_params.model_dump() if req.training_params else None,
        ragas_metrics_config=req.ragas_metrics.model_dump(),
        security_config=req.security_config.model_dump() if req.security_config else None,
        register_model=req.register_model,
        config_snapshot=req.model_dump(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    logger.info("Created run %d for model %s", run.id, run.model_name)
    return run


@app.get("/runs", response_model=list[RunOut])
def list_runs(status: str | None = None, db: Session = Depends(get_db)):
    q = db.query(PipelineRun)
    if status:
        q = q.filter(PipelineRun.status == status)
    return q.order_by(PipelineRun.created_at.desc()).all()


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(PipelineRun).get(run_id)
    if not run:
        raise HTTPException(404, "Run introuvable")
    return run


@app.patch("/runs/{run_id}/status")
def update_run_status(run_id: int, status: str, db: Session = Depends(get_db)):
    run = db.query(PipelineRun).get(run_id)
    if not run:
        raise HTTPException(404, "Run introuvable")
    run.status = RunStatus(status)
    if status in ("completed", "failed"):
        run.finished_at = datetime.datetime.utcnow()
    db.commit()
    return {"ok": True}


@app.patch("/runs/{run_id}/logs")
def append_run_logs(run_id: int, text: str, db: Session = Depends(get_db)):
    run = db.query(PipelineRun).get(run_id)
    if not run:
        raise HTTPException(404, "Run introuvable")
    run.logs = (run.logs or "") + text + "\n"
    db.commit()
    return {"ok": True}


@app.post("/runs/{run_id}/results")
def add_run_results(run_id: int, metrics: dict[str, float], db: Session = Depends(get_db)):
    run = db.query(PipelineRun).get(run_id)
    if not run:
        raise HTTPException(404, "Run introuvable")
    for name, value in metrics.items():
        db.add(RunResult(run_id=run_id, metric_name=name, metric_value=value))
    db.commit()
    return {"ok": True}
