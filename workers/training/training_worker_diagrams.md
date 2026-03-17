# Training Worker - Diagrammes Mermaid

## Diagramme de structure (Class Diagram)

```mermaid
classDiagram
    class WorkerLoop {
      +poll_loop()
      +process_run(run)
    }

    class Helpers {
      +_notify(run_id, status, logs, metrics, error)
      +_notify_status(run_id, status, logs, **kwargs)
      +_log(run_id, message)
      +_resolve_dataset_path(run, ds_type) str
      +_detect_domain(run) str|None
    }

    class LoraUtils {
      +_build_lora_config(params) PeftLoraConfig|None
      +_find_target_modules(model, requested) list~str~
      +KNOWN_LINEAR_NAMES set
    }

    class TrainingPipeline {
      +snapshot_download(model_hf_id)
      +AutoTokenizer.from_pretrained(cache_dir)
      +AutoModelForCausalLM.from_pretrained(cache_dir)
      +Trainer.train()
      +model.merge_and_unload()
    }

    class FastAPI_Backend
    class MLflow_Tracking
    class HuggingFace_Hub
    class HuggingFace_Transformers
    class PEFT_LoRA
    class Datasets_Lib
    class LakehouseResolver

    WorkerLoop --> Helpers
    WorkerLoop --> LoraUtils
    WorkerLoop --> TrainingPipeline

    Helpers --> FastAPI_Backend : status/logs/results
    Helpers --> LakehouseResolver : resolve snapshot path
    TrainingPipeline --> HuggingFace_Hub : model files cache
    TrainingPipeline --> HuggingFace_Transformers : tokenizer/model/trainer
    TrainingPipeline --> PEFT_LoRA : apply LoRA
    TrainingPipeline --> Datasets_Lib : load/tokenize dataset
    WorkerLoop --> MLflow_Tracking : params/metrics/artifacts/registry
```

## Diagramme de sequence

```mermaid
sequenceDiagram
    actor Scheduler
    participant Poll as poll_loop()
    participant API as FastAPI API
    participant Proc as process_run()
    participant HF as HuggingFace
    participant DS as Dataset
    participant TR as Trainer
    participant MLF as MLflow

    Scheduler->>Poll: start worker
    loop Every POLL_INTERVAL seconds
        Poll->>API: GET /runs?status=pending
        API-->>Poll: run list
        Poll->>Poll: filter task_type=finetune

        alt finetune run available
            Poll->>Proc: process_run(run)
            Proc->>API: PATCH status=training
            Proc->>MLF: start_run + set tags

            Proc->>HF: snapshot_download(model_id)
            HF-->>Proc: cache_dir
            Proc->>HF: load tokenizer + config + model
            Proc->>Proc: apply LoRA if configured

            Proc->>DS: resolve train dataset path
            Proc->>DS: load_dataset + tokenize

            Proc->>TR: Trainer.train()
            TR-->>Proc: train_result metrics

            Proc->>Proc: merge LoRA (optional) + save model/tokenizer
            Proc->>MLF: log_metrics + log_artifacts + register model
            Proc->>API: POST /runs/{id}/results (train metrics)
            Proc->>API: PATCH status=evaluating
        else error in process_run
            Proc->>API: PATCH status=failed
            Proc->>API: PATCH logs="ERROR: traceback"
        end
    end
```

## Diagramme d'activite

```mermaid
flowchart TD
    A(["Start poll_loop"]) --> B["GET /runs?status=pending"];
    B --> C["Filter task_type == finetune"];
    C --> D{"Run found?"};

    D -- "No" --> Z["Wait POLL_INTERVAL"];
    Z --> A;
    D -- "Yes" --> E["process_run(run)"];

    E --> F["Notify status=training"];
    F --> G["Start MLflow run + set experiment/tags"];
    G --> H["Download model files (snapshot_download)"];
    H --> I["Load tokenizer + config + model"];
    I --> J{"LoRA configured?"};
    J -- "Yes" --> K["Resolve target modules + apply LoRA"];
    J -- "No" --> L["Skip LoRA"];
    K --> M["Resolve train dataset path"];
    L --> M;

    M --> N["Load JSON dataset"];
    N --> O["Tokenize question/answer samples"];
    O --> P["Build TrainingArguments + Trainer"];
    P --> Q["Run trainer.train()"];
    Q --> R["Compute train metrics + perplexity"];
    R --> S{"LoRA active?"};
    S -- "Yes" --> T["Merge LoRA into base model"];
    S -- "No" --> U["Keep current model"];
    T --> V["Save model + tokenizer to OUTPUT_DIR"];
    U --> V;
    V --> W["Log metrics/artifacts + register model in MLflow"];
    W --> X["POST /runs/{id}/results (train metrics)"];
    X --> Y["PATCH status=evaluating"];
    Y --> A;
```
