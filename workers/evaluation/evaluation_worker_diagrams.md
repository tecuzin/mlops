# Evaluation Worker - Diagrammes Mermaid

## Diagramme de structure (Class Diagram)

```mermaid
classDiagram
    class WorkerLoop {
      +poll_loop()
      +process_run(run)
      +_tag_sync_loop()
    }

    class Helpers {
      +_safe_float(v) float
      +_notify(run_id, status, logs, metrics, error)
      +_notify_status(run_id, status, logs, **kwargs)
      +_log(run_id, message)
      +_resolve_eval_path(run) str
      +_detect_domain(run) str|None
    }

    class Inference {
      +_load_trained_model(model_output_path, run_id)
      +_generate_answers(model, tokenizer, ds, run_id) list~str~
    }

    class Evaluation {
      +_compute_mlscore(scores) float
      +METRIC_MAP dict
      +MLSCORE_WEIGHTS dict
    }

    class TagReconciliation {
      +_reconcile_run_tags(run_data, datasets)
    }

    class FastAPI_Backend
    class MLflow_Tracking
    class RAGAS
    class Mistral_Judge
    class Transformers
    class Datasets_Lib
    class LakehouseResolver

    WorkerLoop --> Helpers
    WorkerLoop --> Inference
    WorkerLoop --> Evaluation
    WorkerLoop --> TagReconciliation

    Helpers --> FastAPI_Backend : status/logs/results
    Helpers --> LakehouseResolver : resolve eval snapshot
    Inference --> Transformers : load model/tokenizer + generate
    Evaluation --> RAGAS : evaluate(dataset, metrics, llm, embeddings)
    Evaluation --> Mistral_Judge : ChatMistralAI + embeddings
    Evaluation --> Datasets_Lib : load eval dataset
    WorkerLoop --> MLflow_Tracking : tags/metrics/artifacts
```

## Diagramme de sequence

```mermaid
sequenceDiagram
    actor Scheduler
    participant Poll as poll_loop()
    participant API as FastAPI API
    participant Proc as process_run()
    participant INF as Inference
    participant RAG as RAGAS
    participant MLF as MLflow
    participant TS as TagSync

    Scheduler->>Poll: start worker
    Scheduler->>TS: start _tag_sync_loop() thread

    loop Every POLL_INTERVAL seconds
        Poll->>API: GET /runs?status=pending
        Poll->>API: GET /runs?status=evaluating
        API-->>Poll: eval_only + awaiting_eval runs

        alt run available
            Poll->>Proc: process_run(run)
            Proc->>API: PATCH status=evaluating
            Proc->>MLF: start_run + set initial tags

            Proc->>API: resolve eval dataset (id or lakehouse ref)
            Proc->>Proc: load_dataset(eval)

            alt finetune model directory exists
                Proc->>INF: _load_trained_model()
                Proc->>INF: _generate_answers()
                INF-->>Proc: generated answers
                Proc->>Proc: replace dataset answers
            else no trained model
                Proc->>Proc: use dataset predefined answers
            end

            Proc->>RAG: evaluate(dataset, selected_metrics, llm, embeddings)
            RAG-->>Proc: per-sample scores
            Proc->>Proc: compute averages + ml_score + validation
            Proc->>MLF: log_metrics + params + tags + optional artifact
            Proc->>API: persist mlflow_run_id
            Proc->>API: POST /runs/{id}/results
            Proc->>API: PATCH status=completed
        else error in process_run
            Proc->>API: PATCH status=failed
            Proc->>API: PATCH logs="ERROR: traceback"
        end
    end

    loop Every TAG_SYNC_INTERVAL seconds
        TS->>API: GET /datasets + GET /runs
        API-->>TS: current state
        TS->>MLF: reconcile lifecycle/domain/validation tags
    end
```

## Diagramme d'activite

```mermaid
flowchart TD
    A(["Start poll_loop"]) --> B["GET /runs?status=pending"];
    B --> C["Keep eval_only runs"];
    C --> D["GET /runs?status=evaluating"];
    D --> E["Merge candidates (eval_only + evaluating)"];
    E --> F{"Run found?"};

    F -- "No" --> Z["Wait POLL_INTERVAL"];
    Z --> A;
    F -- "Yes" --> G["process_run(run)"];

    G --> H["Notify status=evaluating"];
    H --> I["Start MLflow run + initial tags"];
    I --> J["Resolve/load eval dataset"];
    J --> K{"Finetune run with trained model dir?"};
    K -- "Yes" --> L["Load trained model + tokenizer"];
    L --> M["Generate answers for each sample"];
    M --> N["Replace dataset answers with generated answers"];
    K -- "No" --> O["Use dataset answers as-is"];
    N --> P["Select active RAGAS metrics"];
    O --> P;

    P --> Q["Init Mistral judge + embeddings"];
    Q --> R["Run ragas.evaluate(...)"];
    R --> S["Aggregate scores + compute MLScore"];
    S --> T["Set validation tag by threshold"];
    T --> U["Log metrics/params/tags/artifacts to MLflow"];
    U --> V["Persist mlflow_run_id in API"];
    V --> W["POST /runs/{id}/results (scores)"];
    W --> X["PATCH status=completed"];
    X --> A;
```
