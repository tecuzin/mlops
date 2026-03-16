# Schémas UML — Architecture MLOps

Ce document centralise les schémas UML du projet pour faciliter la compréhension globale et les revues d'architecture.

## 1) Diagramme de composants

```mermaid
flowchart LR
    UI[Streamlit UI] -->|POST /runs| API[FastAPI API]
    API --> DB[(PostgreSQL)]
    API --> MLFLOW[MLflow Tracking/Registry]

    TRAIN[Training Worker] -->|GET/PATCH/POST runs| API
    EVAL[Evaluation Worker] -->|GET/PATCH/POST runs| API
    SEC[Security Worker] -->|GET/PATCH/POST runs| API

    TRAIN --> MLFLOW
    EVAL --> MLFLOW
    SEC --> MLFLOW

    SPARK[Spark Service] --> MINIO[(MinIO S3)]
    SPARK --> NESSIE[Nessie Catalog]
    SPARK --> META[[lakehouse/metadata]]

    TRAIN --> META
    EVAL --> META
    SEC --> META
```

## 2) Diagramme de séquence — Run `finetune` (lakehouse snapshot)

```mermaid
sequenceDiagram
    participant User as Utilisateur
    participant UI as Streamlit UI
    participant API as FastAPI
    participant DB as PostgreSQL
    participant TW as Training Worker
    participant EW as Evaluation Worker
    participant ML as MLflow
    participant Meta as Lakehouse Metadata

    User->>UI: Configure modèle + refs lakehouse
    UI->>API: POST /runs (train_lakehouse_ref, eval_lakehouse_ref)
    API->>DB: Insert PipelineRun (status=pending)
    API-->>UI: 201 RunOut

    TW->>API: GET /runs?status=pending
    TW->>Meta: Resolve train_lakehouse_ref(catalog, reference, snapshot_id)
    TW->>API: PATCH status=training
    TW->>ML: start_run + log metrics
    TW->>API: POST /runs/{id}/results (train_loss, perplexity)
    TW->>API: PATCH status=evaluating

    EW->>API: GET /runs?status=evaluating
    EW->>Meta: Resolve eval_lakehouse_ref(catalog, reference, snapshot_id)
    EW->>ML: log RAGAS + mlscore
    EW->>API: POST /runs/{id}/results (faithfulness, ...)
    EW->>API: PATCH status=completed

    UI->>API: GET /runs/{id}
    API-->>UI: RunOut + results
```

## 3) Diagramme d'activités — Pipeline Medallion

```mermaid
flowchart TD
    A[Source JSONL] --> B[Bronze Ingest]
    B --> C[Bronze Parquet]
    C --> D[Silver Transform]
    D --> E{Qualité OK ?}
    E -- Non --> F[Stop + logs qualité]
    E -- Oui --> G[Silver Parquet]
    G --> H[Gold Materialize]
    H --> I[Gold Dataset Ready]
    I --> N[Publication table via Nessie]
    N --> J[Écriture metadata snapshot + catalog_commit_id]
    J --> K[Consommation workers via *_lakehouse_ref]
```

## 4) Diagramme de classes — Modèle de run (simplifié)

```mermaid
classDiagram
    class Dataset {
      +int id
      +string name
      +string file_path
      +string dataset_type
      +int row_count
    }

    class PipelineRun {
      +int id
      +string experiment_name
      +string status
      +string model_name
      +string model_id
      +string task_type
      +json config_snapshot
      +int? train_dataset_id
      +int? eval_dataset_id
      +json? train_lakehouse_ref
      +json? eval_lakehouse_ref
      +string? mlflow_run_id
      +string? mlflow_model_name
      +string? mlflow_model_version
    }

    class RunResult {
      +int id
      +int run_id
      +string metric_name
      +float metric_value
    }

    Dataset "1" <-- "0..*" PipelineRun : train_dataset
    Dataset "1" <-- "0..*" PipelineRun : eval_dataset
    PipelineRun "1" --> "0..*" RunResult : results
```
