"""Create tables and seed initial datasets."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from db.models import Base, Dataset
from db.session import SessionLocal, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED_DATASETS = [
    {
        "name": "rag_qa_train",
        "description": "Jeu d'entraînement QA généraliste (10 exemples)",
        "file_path": "/app/data/train/rag_qa_train.jsonl",
        "dataset_type": "train",
    },
    {
        "name": "medical_qa_train",
        "description": "Jeu d'entraînement QA médical (5 exemples)",
        "file_path": "/app/data/train/medical_qa_train.jsonl",
        "dataset_type": "train",
    },
    {
        "name": "legal_qa_train",
        "description": "Jeu d'entraînement QA juridique (5 exemples)",
        "file_path": "/app/data/train/legal_qa_train.jsonl",
        "dataset_type": "train",
    },
    {
        "name": "ragas_eval",
        "description": "Jeu d'évaluation RAGAS généraliste (8 exemples)",
        "file_path": "/app/data/eval/ragas_eval.jsonl",
        "dataset_type": "eval",
    },
    {
        "name": "medical_ragas_eval",
        "description": "Jeu d'évaluation RAGAS médical (3 exemples)",
        "file_path": "/app/data/eval/medical_ragas_eval.jsonl",
        "dataset_type": "eval",
    },
    {
        "name": "legal_ragas_eval",
        "description": "Jeu d'évaluation RAGAS juridique (3 exemples)",
        "file_path": "/app/data/eval/legal_ragas_eval.jsonl",
        "dataset_type": "eval",
    },
]


def _count_lines(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def seed():
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created")

    db = SessionLocal()
    try:
        for ds_data in SEED_DATASETS:
            exists = db.query(Dataset).filter_by(name=ds_data["name"]).first()
            if exists:
                logger.info("Dataset '%s' already exists, skipping", ds_data["name"])
                continue
            ds = Dataset(
                name=ds_data["name"],
                description=ds_data["description"],
                file_path=ds_data["file_path"],
                dataset_type=ds_data["dataset_type"],
                row_count=_count_lines(ds_data["file_path"]),
            )
            db.add(ds)
            logger.info("Seeded dataset '%s'", ds_data["name"])
        db.commit()
        logger.info("Seed complete")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
