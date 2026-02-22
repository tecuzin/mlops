# Spécification — Équivalent AWS-Native du projet MLOps

> Transposition complète de l'architecture MLOps (FastAPI, Streamlit, MLflow, PostgreSQL, workers, RabbitMQ, Digital.ai Release) vers une stack 100% AWS.

---

## Table des matières

1. [Matrice de correspondance](#1-matrice-de-correspondance)
2. [Architecture AWS cible](#2-architecture-aws-cible)
3. [Amazon SageMaker — Training et Model Registry](#3-amazon-sagemaker--training-et-model-registry)
4. [AWS App Runner — API et UI](#4-aws-app-runner--api-et-ui)
5. [Amazon EventBridge — Orchestration événementielle](#5-amazon-eventbridge--orchestration-événementielle)
6. [Amazon SQS — Files de traitement](#6-amazon-sqs--files-de-traitement)
7. [AWS Lambda — Evaluation Worker](#7-aws-lambda--evaluation-worker)
8. [AWS Step Functions — Orchestration de release](#8-aws-step-functions--orchestration-de-release)
9. [Amazon Bedrock — LLM juge pour RAGAS](#9-amazon-bedrock--llm-juge-pour-ragas)
10. [Amazon SNS + AWS Chatbot — Notifications Teams](#10-amazon-sns--aws-chatbot--notifications-teams)
11. [Amazon RDS for PostgreSQL](#11-amazon-rds-for-postgresql)
12. [Amazon S3 — Données et artefacts](#12-amazon-s3--données-et-artefacts)
13. [AWS Secrets Manager — Secrets](#13-aws-secrets-manager--secrets)
14. [Amazon CloudWatch — Observabilité](#14-amazon-cloudwatch--observabilité)
15. [Sécurité et identité (IAM)](#15-sécurité-et-identité-iam)
16. [Infrastructure as Code (AWS CDK)](#16-infrastructure-as-code-aws-cdk)
17. [Workflow de bout en bout](#17-workflow-de-bout-en-bout)
18. [Estimation des coûts](#18-estimation-des-coûts)
19. [Plan de migration](#19-plan-de-migration)

---

## 1. Matrice de correspondance

| Composant actuel | Service AWS | Configuration recommandée | Justification |
|---|---|---|---|
| **PostgreSQL 16** (Docker) | Amazon RDS for PostgreSQL | db.t4g.micro (2 vCPU, 1 GiB) | PaaS managé, Multi-AZ, backup auto |
| **FastAPI** (Docker) | AWS App Runner | 0.25 vCPU / 0.5 GB | Déploiement depuis ECR, HTTPS natif, auto-scale |
| **Streamlit UI** (Docker) | AWS App Runner | 0.25 vCPU / 0.5 GB | Même stack que l'API, domaine custom |
| **MLflow Tracking** (Docker) | Amazon SageMaker Experiments | (inclus dans SageMaker) | Tracking natif, intégration S3, SDK Python |
| **MLflow Model Registry** | SageMaker Model Registry | (inclus dans SageMaker) | Model Groups, versions, approval statuses |
| **Training Worker** (Docker, poll) | SageMaker Training Jobs | ml.g4dn.xlarge (GPU) ou ml.m5.xlarge (CPU) | Instances éphémères, scale-to-zero, Spot disponible |
| **Evaluation Worker** (Docker, poll) | AWS Lambda + ECS Fargate Task | Lambda (1 GB, 15 min) ou Fargate (1 vCPU, 2 GB) | Event-driven via SQS, serverless |
| **RabbitMQ** (Docker) | Amazon EventBridge + Amazon SQS | Standard | Bus d'événements + files de traitement |
| **Digital.ai Release** | AWS Step Functions | Standard | Machine à états, approbations humaines, branching |
| **Teams notifications** | Amazon SNS + AWS Chatbot | Standard | Connecteur Teams/Slack natif AWS |
| **Mistral API** (LLM juge RAGAS) | Amazon Bedrock | Mistral Large / Small | Serverless, pay-per-token, données dans AWS |
| **HuggingFace Hub** | SageMaker HuggingFace Containers + Model Hub | — | Images Deep Learning pré-construites |
| **Volumes Docker** | Amazon S3 | S3 Standard | Stockage objets pour modèles et datasets |
| **`.env` / Secrets** | AWS Secrets Manager | Standard | Rotation auto, audit CloudTrail, IAM |
| **Monitoring** | Amazon CloudWatch + AWS X-Ray | — | Métriques, logs, traces, alertes, dashboards |
| **Container Registry** | Amazon ECR | Private | Stockage des images Docker |
| **IaC** | AWS CDK (TypeScript) / CloudFormation | — | Déploiement reproductible |

---

## 2. Architecture AWS cible

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                  AWS ACCOUNT                                     │
│                               VPC: vpc-mlops                                     │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                          PUBLIC SUBNETS (2 AZ)                              │ │
│  │                                                                             │ │
│  │   ┌──────────────┐     ┌──────────────┐                                    │ │
│  │   │  App Runner  │     │  App Runner  │                                    │ │
│  │   │  Streamlit   │────►│   FastAPI    │                                    │ │
│  │   │  (HTTPS)     │     │  (HTTPS)     │                                    │ │
│  │   └──────────────┘     └──────┬───────┘                                    │ │
│  │                               │                                             │ │
│  └───────────────────────────────┼─────────────────────────────────────────────┘ │
│                                  │ publish                                       │
│                                  ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Amazon EventBridge                                  │ │
│  │                      Bus: mlops-events                                      │ │
│  │                                                                             │ │
│  │   Rules:                                                                    │ │
│  │     run.created ──► SQS (training) + Step Functions (release)               │ │
│  │     run.training_done ──► SQS (evaluation)                                  │ │
│  │     run.eval_passed ──► SNS (Teams notification)                            │ │
│  │     run.eval_rejected ──► SNS (Teams notification)                          │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                  │                                               │
│               ┌──────────────────┼──────────────────┐                            │
│               ▼                  ▼                  ▼                            │
│  ┌──────────────────┐ ┌──────────────────┐ ┌───────────────────┐                │
│  │   SageMaker      │ │  ECS Fargate     │ │  Step Functions   │                │
│  │   Training Job   │ │  Task (Eval)     │ │  (Release         │                │
│  │   (GPU)          │ │  ou Lambda       │ │   Workflow)       │                │
│  └────────┬─────────┘ └────────┬─────────┘ └─────────┬─────────┘                │
│           │                    │                      │                           │
│  ┌────────┼────────────────────┼──────────────────────┼─────────────────────────┐ │
│  │        │              PRIVATE SUBNETS (2 AZ)       │                         │ │
│  │        ▼                    ▼                      ▼                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │  SageMaker   │  │  Amazon      │  │   Amazon S3  │  │   AWS        │    │ │
│  │  │  (Tracking   │  │  RDS         │  │  (datasets   │  │  Secrets     │    │ │
│  │  │   + Model    │  │  PostgreSQL  │  │   + modèles) │  │  Manager     │    │ │
│  │  │   Registry)  │  │  Multi-AZ    │  │              │  │              │    │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│  │  Amazon Bedrock  │  │  Amazon SNS +    │  │  Amazon          │               │
│  │  (Mistral on     │  │  AWS Chatbot     │  │  CloudWatch      │               │
│  │   Bedrock)       │  │  (→ Teams)       │  │  + X-Ray         │               │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘               │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │  Amazon ECR (images Docker)  │  AWS CDK (Infrastructure as Code)            │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Amazon SageMaker — Training et Model Registry

Remplace MLflow Tracking Server + MLflow Model Registry + le training worker Docker.

### 3.1. SageMaker Experiments (Tracking)

| Fonctionnalité MLflow | Équivalent SageMaker |
|---|---|
| `mlflow.set_experiment()` | `Experiment.create(experiment_name=...)` |
| `mlflow.start_run()` | `Run.create(experiment_name=..., run_name=...)` |
| `mlflow.log_metrics()` | `run.log_metric(name, value, step)` |
| `mlflow.log_artifacts()` | `run.log_artifact(name, value, is_input=False)` |
| `mlflow.set_tag()` | `run.log_parameter(name, value)` |
| MLflow UI (`http://mlflow:5000`) | SageMaker Studio (`https://studio.{region}.sagemaker.aws`) |

SageMaker expose également un **endpoint MLflow compatible**, permettant de conserver le code MLflow existant :

```python
import mlflow

mlflow.set_tracking_uri(f"arn:aws:sagemaker:{region}:{account_id}:mlflow-tracking-server/mlops")

mlflow.set_experiment("rag-qa-finetune")
with mlflow.start_run(run_name="mistral-7b-rag-qa"):
    mlflow.log_metrics({"train_loss": 0.42, "perplexity": 1.52})
    mlflow.log_artifacts("./model", "model")
```

> Note : Depuis 2024, SageMaker supporte nativement un MLflow Tracking Server managé. Le code des workers peut rester quasi-identique.

### 3.2. SageMaker Model Registry

Remplace le MLflow Model Registry avec un système de **Model Package Groups** et **Model Packages** :

| Fonctionnalité MLflow | Équivalent SageMaker |
|---|---|
| Registered Model | Model Package Group |
| Model Version | Model Package (dans le groupe) |
| Stage `Production` / `Archived` | Approval Status: `Approved` / `Rejected` / `PendingManualApproval` |
| Tags sur la version | Tags sur le Model Package |
| `create_registered_model()` | `create_model_package_group()` |
| `create_model_version()` | `create_model_package()` |
| `set_model_version_tag()` | `add_tags()` sur le Model Package ARN |

Exemple d'enregistrement :

```python
import boto3

sm = boto3.client("sagemaker")

sm.create_model_package_group(
    ModelPackageGroupName="mistral-7b-rag-qa",
    ModelPackageGroupDescription="Fine-tuned Mistral 7B for RAG QA",
    Tags=[
        {"Key": "project", "Value": "mlops"},
        {"Key": "domain", "Value": "medic"},
    ],
)

sm.create_model_package(
    ModelPackageGroupName="mistral-7b-rag-qa",
    ModelPackageDescription="v1 — LoRA r=16, 3 epochs",
    InferenceSpecification={
        "Containers": [{
            "Image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/mlops-inference:latest",
            "ModelDataUrl": "s3://mlops-model-outputs/mistral-7b-rag-qa/model.tar.gz",
        }],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="PendingManualApproval",
    AdditionalInferenceSpecifications=[],
    Tags=[
        {"Key": "lifecycle", "Value": "finetuned"},
        {"Key": "validation", "Value": "pending"},
        {"Key": "ml_score_threshold", "Value": "0.7"},
    ],
)
```

### 3.3. Transitions de statut du modèle

| Situation | MLflow (actuel) | SageMaker Model Registry |
|---|---|---|
| Modèle enregistré après training | Tag `lifecycle=finetuned` | `ModelApprovalStatus = PendingManualApproval` |
| Modèle validé (MLScore ≥ seuil + approbation humaine) | Tag `deployment_status=go_prod`, Stage `Production` | `ModelApprovalStatus = Approved`, Tag `go_prod` |
| Modèle rejeté (MLScore < seuil) | Tag `deployment_status=rejected`, Stage `Archived` | `ModelApprovalStatus = Rejected`, Tag `rejected` |

```python
sm.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",
    ApprovalDescription="MLScore 0.82 ≥ seuil 0.7 — approuvé par l'équipe ML",
)

sm.add_tags(
    ResourceArn=model_package_arn,
    Tags=[
        {"Key": "deployment_status", "Value": "go_prod"},
        {"Key": "ml_score", "Value": "0.82"},
    ],
)
```

### 3.4. SageMaker Training Jobs

Remplacent le training worker Docker. Instances éphémères avec GPU, facturées à la seconde :

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point="worker.py",
    source_dir="workers/training",
    instance_type="ml.g4dn.xlarge",      # 1x T4 GPU, 4 vCPU, 16 GB
    instance_count=1,
    role=sagemaker_execution_role,
    transformers_version="4.44",
    pytorch_version="2.4",
    py_version="py311",
    hyperparameters={
        "run_id": "42",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    environment={
        "API_URL": "https://api.mlops.example.com",
        "MLFLOW_TRACKING_URI": mlflow_tracking_arn,
    },
    output_path="s3://mlops-model-outputs/",
    use_spot_instances=True,              # Jusqu'à 90% d'économies
    max_wait=7200,
    max_run=3600,
    tags=[
        {"Key": "project", "Value": "mlops"},
        {"Key": "run_id", "Value": "42"},
    ],
)

huggingface_estimator.fit({
    "train": "s3://mlops-datasets/train/rag_qa_train.jsonl",
})
```

| Aspect | Docker (actuel) | SageMaker Training Job (cible) |
|---|---|---|
| Déclenchement | Polling HTTP toutes les 10s | Événement EventBridge → Step Functions → `CreateTrainingJob` |
| Exécution | Container permanent (8 GB réservé) | Instance éphémère GPU (scale-to-zero) |
| GPU | Non garanti, réservation mémoire | Instance dédiée ml.g4dn.xlarge (T4) ou ml.p3.2xlarge (V100) |
| Spot instances | Non disponible | Jusqu'à 90% d'économies |
| Stockage modèle | Volume Docker `model_outputs` | Amazon S3 |
| MLflow | `http://mlflow:5000` | MLflow managé SageMaker ou SageMaker Experiments |
| Coût | Container 24/7 | Facturation à la seconde, uniquement pendant le training |

---

## 4. AWS App Runner — API et UI

Remplace les containers Docker FastAPI et Streamlit.

### 4.1. Pourquoi App Runner

| Critère | ECS Fargate | App Runner |
|---|---|---|
| Configuration | Task Definition + Service + ALB + Target Group | Image ECR → déployé |
| HTTPS | ALB + ACM | Inclus |
| Auto-scaling | Configuration manuelle | Automatique (basé sur concurrency) |
| Déploiement | Blue/Green via CodeDeploy | Rolling automatique |
| Complexité | Élevée | Minimale |
| VPC | Natif | VPC Connector (pour accéder RDS, etc.) |

### 4.2. App Runner : API FastAPI

| Propriété | Valeur |
|---|---|
| Nom | `mlops-api` |
| Source | ECR : `{account}.dkr.ecr.{region}.amazonaws.com/mlops-api:latest` |
| CPU / Mémoire | 0.5 vCPU / 1 GB |
| Port | 8000 |
| Auto-scaling | Min 1, Max 5 (concurrency: 50) |
| VPC Connector | `mlops-vpc-connector` (accès RDS, S3 via VPC endpoint) |
| Instance Role | `mlops-api-role` (accès Secrets Manager, EventBridge, S3) |

### 4.3. App Runner : UI Streamlit

| Propriété | Valeur |
|---|---|
| Nom | `mlops-ui` |
| Source | ECR : `{account}.dkr.ecr.{region}.amazonaws.com/mlops-ui:latest` |
| CPU / Mémoire | 0.25 vCPU / 0.5 GB |
| Port | 8501 |
| Auto-scaling | Min 1, Max 3 |
| Variable d'env | `API_URL=https://mlops-api.{id}.{region}.apprunner.aws` |

### 4.4. Modification de l'API pour EventBridge

L'API publie des événements dans EventBridge au lieu de RabbitMQ :

```python
import boto3
import json
import os
from datetime import datetime

eventbridge = boto3.client("events")
EVENT_BUS = os.getenv("EVENT_BUS_NAME", "mlops-events")

def publish_event(event_type: str, run_id: int, payload: dict):
    eventbridge.put_events(
        Entries=[{
            "Source": "mlops.api",
            "DetailType": event_type,
            "Detail": json.dumps({
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload,
            }),
            "EventBusName": EVENT_BUS,
        }]
    )
```

Intégration dans l'endpoint `PATCH /runs/{run_id}/status` :

```python
@app.patch("/runs/{run_id}/status")
def update_run_status(run_id: int, status: str, db: Session = Depends(get_db)):
    run = db.query(PipelineRun).get(run_id)
    if not run:
        raise HTTPException(404, "Run introuvable")
    run.status = RunStatus(status)
    if status in ("completed", "failed"):
        run.finished_at = datetime.utcnow()
    db.commit()

    publish_event(f"run.{status}", run_id=run.id, payload={
        "model_name": run.model_name,
        "model_id": run.model_id,
        "task_type": run.task_type,
        "experiment_name": run.experiment_name,
    })
    return {"ok": True}
```

Et dans l'endpoint `POST /runs` (création initiale) :

```python
@app.post("/runs", response_model=RunOut, status_code=201)
def create_run(req: RunCreateRequest, db: Session = Depends(get_db)):
    # ... validation et création ...
    db.commit()
    db.refresh(run)

    publish_event("run.created", run_id=run.id, payload={
        "model_name": run.model_name,
        "model_id": run.model_id,
        "task_type": run.task_type,
        "experiment_name": run.experiment_name,
    })
    return run
```

---

## 5. Amazon EventBridge — Orchestration événementielle

Remplace RabbitMQ comme bus d'événements central.

### 5.1. Pourquoi EventBridge plutôt que SQS seul

| Critère | Amazon SQS seul | Amazon EventBridge |
|---|---|---|
| Pattern | Point-à-point (1 producteur → 1 consommateur) | Fan-out natif (1 événement → N cibles) |
| Routage | Pas de filtrage natif | Règles de filtrage par contenu |
| Cibles | Application (poll) | Lambda, Step Functions, SQS, SNS, API Gateway... |
| Schema | Libre | Schema Registry optionnel |
| Replay | Non | Oui (archive + replay jusqu'à 90 jours) |
| Observabilité | Métriques basiques | CloudWatch Metrics + logs détaillés |

EventBridge sert de **routeur intelligent** : chaque événement est dispatché automatiquement vers les bonnes cibles selon des règles de filtrage.

### 5.2. Event Bus et règles

| Règle | Pattern de filtrage | Cible(s) | Description |
|---|---|---|---|
| `rule-run-created-training` | `{"detail-type": ["run.created"], "detail": {"payload": {"task_type": ["finetune"]}}}` | SQS `mlops-training-queue` | Déclenche le training |
| `rule-run-created-eval-only` | `{"detail-type": ["run.created"], "detail": {"payload": {"task_type": ["eval_only"]}}}` | SQS `mlops-eval-queue` | Déclenche l'évaluation directe |
| `rule-run-created-release` | `{"detail-type": ["run.created"]}` | Step Functions `mlops-release-workflow` | Démarre le workflow de release |
| `rule-training-done` | `{"detail-type": ["run.training_done"]}` | SQS `mlops-eval-queue` | Déclenche l'évaluation post-training |
| `rule-eval-passed` | `{"detail-type": ["run.eval_passed"]}` | SNS `mlops-notifications` | Notifie Teams (modèle prêt) |
| `rule-eval-rejected` | `{"detail-type": ["run.eval_rejected"]}` | SNS `mlops-notifications` | Notifie Teams (modèle rejeté) |

### 5.3. Format des événements

```json
{
  "version": "0",
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "source": "mlops.api",
  "detail-type": "run.created",
  "time": "2026-02-22T14:30:00Z",
  "region": "eu-west-1",
  "detail": {
    "run_id": 42,
    "timestamp": "2026-02-22T14:30:00Z",
    "payload": {
      "model_name": "mistral-7b-rag-qa",
      "model_id": "mistralai/Mistral-7B-v0.1",
      "task_type": "finetune",
      "experiment_name": "rag-qa-finetune"
    }
  }
}
```

### 5.4. Archive et replay

EventBridge peut archiver tous les événements et les rejouer en cas de besoin (debug, re-processing) :

```python
eventbridge.create_archive(
    ArchiveName="mlops-events-archive",
    EventSourceArn=f"arn:aws:events:{region}:{account}:event-bus/mlops-events",
    RetentionDays=90,
)
```

### 5.5. Comparaison RabbitMQ vs EventBridge + SQS

| Critère | RabbitMQ (Docker) | EventBridge + SQS |
|---|---|---|
| Déploiement | Container à gérer | 100% serverless |
| Fan-out | Exchange topic + bindings | Règles EventBridge (déclaratif) |
| Dead-letter | Plugin | Natif (SQS DLQ) |
| Filtrage | Routing keys (string) | Patterns JSON (contenu du message) |
| Replay | Non | Archive EventBridge (jusqu'à 90 jours) |
| Intégration AWS | Aucune | Cibles natives (Lambda, Step Functions, SQS, SNS...) |
| Observabilité | Management UI | CloudWatch Metrics + X-Ray |
| Coût | Gratuit (self-hosted) | ~1$/million d'événements |
| Haute disponibilité | Manuel | Incluse (multi-AZ) |

---

## 6. Amazon SQS — Files de traitement

SQS sert de buffer entre EventBridge et les workers de training/évaluation.

### 6.1. Files de messages

| Queue | Visibility Timeout | Retention | DLQ | Consommateur |
|---|---|---|---|---|
| `mlops-training-queue` | 3600s (1h) | 7 jours | `mlops-training-dlq` | Dispatcher Lambda → SageMaker Job |
| `mlops-eval-queue` | 900s (15 min) | 7 jours | `mlops-eval-dlq` | Lambda (évaluation RAGAS) ou ECS Task |
| `mlops-training-dlq` | 30s | 14 jours | — | Monitoring / re-processing manuel |
| `mlops-eval-dlq` | 30s | 14 jours | — | Monitoring / re-processing manuel |

### 6.2. Dispatcher Lambda (Training)

Une Lambda légère consomme les messages de `mlops-training-queue` et soumet des SageMaker Training Jobs :

```python
import boto3
import json

sm = boto3.client("sagemaker")

def lambda_handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        detail = json.loads(body["detail"]) if isinstance(body.get("detail"), str) else body.get("detail", body)
        run_id = detail["run_id"]
        payload = detail["payload"]

        sm.create_training_job(
            TrainingJobName=f"mlops-train-{run_id}-{int(time.time())}",
            AlgorithmSpecification={
                "TrainingImage": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/mlops-training:latest",
                "TrainingInputMode": "File",
            },
            RoleArn=SAGEMAKER_ROLE,
            InputDataConfig=[{
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://mlops-datasets/train/",
                    }
                },
            }],
            OutputDataConfig={
                "S3OutputPath": f"s3://mlops-model-outputs/{payload['model_name']}/",
            },
            ResourceConfig={
                "InstanceType": "ml.g4dn.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            HyperParameters={
                "run_id": str(run_id),
                "model_id": payload["model_id"],
                "experiment_name": payload["experiment_name"],
            },
            Tags=[
                {"Key": "project", "Value": "mlops"},
                {"Key": "run_id", "Value": str(run_id)},
            ],
        )
```

---

## 7. AWS Lambda — Evaluation Worker

Remplace l'evaluation worker Docker pour les évaluations RAGAS.

### 7.1. Choix Lambda vs ECS Fargate

| Critère | Lambda | ECS Fargate Task |
|---|---|---|
| Timeout max | 15 min | Illimité |
| Mémoire max | 10 GB | 30 GB |
| Déclencheur SQS | Natif | Via EventBridge + ECS RunTask |
| Cold start | 1–5s | 30–60s |
| Coût (10 runs/mois) | ~0.50 € | ~2 € |
| Packaging | Container (ECR) ou ZIP | Container (ECR) |

**Recommandation** : utiliser Lambda si l'évaluation RAGAS prend moins de 15 minutes. Sinon, utiliser ECS Fargate Task.

### 7.2. Lambda d'évaluation

```python
import boto3
import json
import os
import httpx
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

bedrock = boto3.client("bedrock-runtime")
API_URL = os.getenv("API_URL")
MLSCORE_THRESHOLD = float(os.getenv("MLSCORE_THRESHOLD", "0.7"))

MLSCORE_WEIGHTS = {
    "faithfulness": 0.30,
    "answer_relevancy": 0.20,
    "context_precision": 0.25,
    "context_recall": 0.25,
}

def lambda_handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        detail = body.get("detail", body)
        run_id = detail["run_id"]

        with httpx.Client(base_url=API_URL, timeout=30) as client:
            client.patch(f"/runs/{run_id}/status", params={"status": "evaluating"})

        # ... chargement dataset, inférence, évaluation RAGAS ...
        # ... (même logique que le worker actuel, adapté pour Bedrock) ...

        mlscore = compute_mlscore(scores)

        event_type = "run.eval_passed" if mlscore >= MLSCORE_THRESHOLD else "run.eval_rejected"
        eventbridge = boto3.client("events")
        eventbridge.put_events(Entries=[{
            "Source": "mlops.evaluation",
            "DetailType": event_type,
            "Detail": json.dumps({
                "run_id": run_id,
                "ml_score": mlscore,
                "scores": scores,
            }),
            "EventBusName": "mlops-events",
        }])
```

### 7.3. Alternative ECS Fargate (évaluations longues)

Pour les évaluations de plus de 15 minutes, utiliser un ECS Fargate Task déclenché par EventBridge :

```python
ecs = boto3.client("ecs")

ecs.run_task(
    cluster="mlops-cluster",
    taskDefinition="mlops-eval-task:latest",
    launchType="FARGATE",
    networkConfiguration={
        "awsvpcConfiguration": {
            "subnets": ["subnet-private-1", "subnet-private-2"],
            "securityGroups": ["sg-eval-worker"],
            "assignPublicIp": "DISABLED",
        }
    },
    overrides={
        "containerOverrides": [{
            "name": "eval-worker",
            "environment": [
                {"name": "RUN_ID", "value": str(run_id)},
            ],
        }],
    },
    tags=[{"key": "run_id", "value": str(run_id)}],
)
```

---

## 8. AWS Step Functions — Orchestration de release

Remplace Digital.ai Release / Azure DevOps Pipelines.

### 8.1. Pourquoi Step Functions

| Critère | AWS CodePipeline | AWS Step Functions |
|---|---|---|
| Modèle | Pipeline linéaire (source → build → deploy) | Machine à états (branching, boucles, parallélisme) |
| Approbation humaine | Manual approval stage (basique) | Callback pattern avec token (flexible) |
| Branching conditionnel | Limité | Natif (`Choice` state) |
| Intégration SageMaker | Via CodeBuild | SDK integration native |
| Intégration EventBridge | Limité | Natif (cible directe) |
| Timeout | 20 jours max par stage | 1 an max par exécution |
| Coût | Gratuit (1 pipeline) | ~0.025$/1000 transitions |
| Visibilité | Console CodePipeline | Graph visuel interactif |

### 8.2. Machine à états (ASL — Amazon States Language)

```json
{
  "Comment": "MLOps Release Workflow — Fine-tuning, Evaluation, Approval",
  "StartAt": "NotifyLoading",
  "States": {

    "NotifyLoading": {
      "Type": "Task",
      "Resource": "arn:aws:states:::http:invoke",
      "Parameters": {
        "Method": "PATCH",
        "Url.$": "States.Format('${ApiUrl}/runs/{}/status?status=loading', $.detail.run_id)",
        "Authentication": { "ConnectionArn": "${ApiConnectionArn}" }
      },
      "Next": "RegisterBaseModel"
    },

    "RegisterBaseModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createModelPackage",
      "Parameters": {
        "ModelPackageGroupName.$": "$.detail.payload.model_name",
        "ModelApprovalStatus": "PendingManualApproval",
        "Tags": [
          { "Key": "lifecycle", "Value": "new" },
          { "Key": "run_id", "Value.$": "States.Format('{}', $.detail.run_id)" }
        ]
      },
      "ResultPath": "$.modelPackage",
      "Next": "CheckTaskType"
    },

    "CheckTaskType": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.detail.payload.task_type",
          "StringEquals": "finetune",
          "Next": "StartTraining"
        }
      ],
      "Default": "StartEvaluation"
    },

    "StartTraining": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "States.Format('mlops-train-{}-{}', $.detail.run_id, $$.Execution.Name)",
        "AlgorithmSpecification": {
          "TrainingImage": "${TrainingImage}",
          "TrainingInputMode": "File"
        },
        "RoleArn": "${SageMakerRoleArn}",
        "InputDataConfig": [{
          "ChannelName": "train",
          "DataSource": {
            "S3DataSource": {
              "S3DataType": "S3Prefix",
              "S3Uri": "s3://mlops-datasets/train/"
            }
          }
        }],
        "OutputDataConfig": {
          "S3OutputPath.$": "States.Format('s3://mlops-model-outputs/{}/', $.detail.payload.model_name)"
        },
        "ResourceConfig": {
          "InstanceType": "ml.g4dn.xlarge",
          "InstanceCount": 1,
          "VolumeSizeInGB": 50
        },
        "StoppingCondition": { "MaxRuntimeInSeconds": 3600 },
        "HyperParameters": {
          "run_id.$": "States.Format('{}', $.detail.run_id)",
          "model_id.$": "$.detail.payload.model_id"
        }
      },
      "ResultPath": "$.trainingResult",
      "Next": "NotifyTrainingDone",
      "Catch": [{
        "ErrorEquals": ["States.ALL"],
        "Next": "TrainingFailed"
      }]
    },

    "NotifyTrainingDone": {
      "Type": "Task",
      "Resource": "arn:aws:states:::events:putEvents",
      "Parameters": {
        "Entries": [{
          "Source": "mlops.stepfunctions",
          "DetailType": "run.training_done",
          "Detail": {
            "run_id.$": "$.detail.run_id",
            "payload.$": "$.detail.payload"
          },
          "EventBusName": "mlops-events"
        }]
      },
      "Next": "StartEvaluation"
    },

    "StartEvaluation": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke.waitForTaskToken",
      "Parameters": {
        "FunctionName": "${EvalDispatcherLambda}",
        "Payload": {
          "run_id.$": "$.detail.run_id",
          "task_token.$": "$$.Task.Token"
        }
      },
      "ResultPath": "$.evalResult",
      "TimeoutSeconds": 3600,
      "Next": "CheckMLScore",
      "Catch": [{
        "ErrorEquals": ["States.ALL"],
        "Next": "EvaluationFailed"
      }]
    },

    "CheckMLScore": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.evalResult.passed",
          "BooleanEquals": true,
          "Next": "NotifyTeamsApproval"
        }
      ],
      "Default": "AutoReject"
    },

    "AutoReject": {
      "Type": "Parallel",
      "Branches": [
        {
          "StartAt": "TagModelRejected",
          "States": {
            "TagModelRejected": {
              "Type": "Task",
              "Resource": "arn:aws:states:::sagemaker:addTags",
              "Parameters": {
                "ResourceArn.$": "$.modelPackage.ModelPackageArn",
                "Tags": [
                  { "Key": "deployment_status", "Value": "rejected" },
                  { "Key": "validation", "Value": "rejected" }
                ]
              },
              "End": true
            }
          }
        },
        {
          "StartAt": "NotifyTeamsRejected",
          "States": {
            "NotifyTeamsRejected": {
              "Type": "Task",
              "Resource": "arn:aws:states:::sns:publish",
              "Parameters": {
                "TopicArn": "${NotificationTopicArn}",
                "Subject": "MLOps — Modèle rejeté",
                "Message.$": "States.Format('Modèle {} rejeté — MLScore {} < seuil 0.7', $.detail.payload.model_name, $.evalResult.ml_score)"
              },
              "End": true
            }
          }
        }
      ],
      "Next": "ReleaseComplete"
    },

    "NotifyTeamsApproval": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "${NotificationTopicArn}",
        "Subject": "MLOps — Modèle prêt à tester",
        "Message.$": "States.Format('Le modèle {} a obtenu un MLScore de {}. Approbation requise.', $.detail.payload.model_name, $.evalResult.ml_score)"
      },
      "Next": "WaitForHumanApproval"
    },

    "WaitForHumanApproval": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke.waitForTaskToken",
      "Parameters": {
        "FunctionName": "${ApprovalHandlerLambda}",
        "Payload": {
          "run_id.$": "$.detail.run_id",
          "model_name.$": "$.detail.payload.model_name",
          "ml_score.$": "$.evalResult.ml_score",
          "task_token.$": "$$.Task.Token"
        }
      },
      "ResultPath": "$.approvalResult",
      "TimeoutSeconds": 259200,
      "Next": "CheckApproval",
      "Catch": [{
        "ErrorEquals": ["States.Timeout"],
        "Next": "ApprovalTimeout"
      }]
    },

    "CheckApproval": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.approvalResult.approved",
          "BooleanEquals": true,
          "Next": "TagModelApproved"
        }
      ],
      "Default": "TagModelManualReject"
    },

    "TagModelApproved": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:updateModelPackage",
      "Parameters": {
        "ModelPackageArn.$": "$.modelPackage.ModelPackageArn",
        "ModelApprovalStatus": "Approved",
        "ApprovalDescription": "Approuvé manuellement après validation"
      },
      "Next": "AddGoProdTag"
    },

    "AddGoProdTag": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:addTags",
      "Parameters": {
        "ResourceArn.$": "$.modelPackage.ModelPackageArn",
        "Tags": [
          { "Key": "deployment_status", "Value": "go_prod" },
          { "Key": "validation", "Value": "validated" }
        ]
      },
      "Next": "NotifyTeamsProduction"
    },

    "NotifyTeamsProduction": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "${NotificationTopicArn}",
        "Subject": "MLOps — Modèle en production",
        "Message.$": "States.Format('Le modèle {} est tagué go_prod et prêt pour le déploiement.', $.detail.payload.model_name)"
      },
      "Next": "ReleaseComplete"
    },

    "TagModelManualReject": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:updateModelPackage",
      "Parameters": {
        "ModelPackageArn.$": "$.modelPackage.ModelPackageArn",
        "ModelApprovalStatus": "Rejected",
        "ApprovalDescription": "Rejeté manuellement lors de la validation"
      },
      "Next": "ReleaseComplete"
    },

    "TrainingFailed": {
      "Type": "Task",
      "Resource": "arn:aws:states:::http:invoke",
      "Parameters": {
        "Method": "PATCH",
        "Url.$": "States.Format('${ApiUrl}/runs/{}/status?status=failed', $.detail.run_id)"
      },
      "Next": "ReleaseComplete"
    },

    "EvaluationFailed": {
      "Type": "Task",
      "Resource": "arn:aws:states:::http:invoke",
      "Parameters": {
        "Method": "PATCH",
        "Url.$": "States.Format('${ApiUrl}/runs/{}/status?status=failed', $.detail.run_id)"
      },
      "Next": "ReleaseComplete"
    },

    "ApprovalTimeout": {
      "Type": "Pass",
      "Result": "Approval timed out after 72h",
      "Next": "TagModelManualReject"
    },

    "ReleaseComplete": {
      "Type": "Succeed"
    }
  }
}
```

### 8.3. Schéma visuel de la machine à états

```
                        ┌──────────────────┐
                        │  NotifyLoading   │
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │RegisterBaseModel │
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │ CheckTaskType    │
                        └───┬──────────┬───┘
                  finetune  │          │  eval_only
                            ▼          ▼
                ┌───────────────┐      │
                │ StartTraining │      │
                │ (SageMaker    │      │
                │  .sync)       │      │
                └───────┬───────┘      │
                        ▼              │
               ┌─────────────────┐     │
               │NotifyTrainingDone│     │
               └────────┬────────┘     │
                        │              │
                        ▼◄─────────────┘
                ┌───────────────┐
                │StartEvaluation│
                │ (Lambda +     │
                │  TaskToken)   │
                └───────┬───────┘
                        ▼
                ┌───────────────┐
                │ CheckMLScore  │
                └───┬───────┬───┘
             passed │       │ failed
                    ▼       ▼
     ┌────────────────┐  ┌──────────────┐
     │NotifyTeams     │  │ AutoReject   │
     │Approval        │  │ (Parallel)   │
     └───────┬────────┘  │ ├─TagRejected│
             ▼           │ └─NotifyTeams│
     ┌────────────────┐  └──────┬───────┘
     │WaitForHuman    │         │
     │Approval        │         │
     │(TaskToken,72h) │         │
     └───┬────────┬───┘         │
  approved│       │rejected     │
          ▼       ▼             │
  ┌────────────┐ ┌──────────┐  │
  │TagApproved │ │TagManual │  │
  │ + go_prod  │ │Reject    │  │
  └─────┬──────┘ └────┬─────┘  │
        ▼              │        │
  ┌──────────────┐     │        │
  │NotifyTeams   │     │        │
  │Production    │     │        │
  └──────┬───────┘     │        │
         │             │        │
         ▼             ▼        ▼
        ┌──────────────────────────┐
        │     ReleaseComplete      │
        └──────────────────────────┘
```

### 8.4. Gate manuelle avec Callback Pattern

Step Functions utilise le **Task Token pattern** pour les approbations humaines. Quand le workflow atteint `WaitForHumanApproval`, il s'arrête et génère un token. La Lambda d'approbation expose une API (ou une interface) pour que l'approbateur puisse approuver ou rejeter :

```python
import boto3

sfn = boto3.client("stepfunctions")

def approve_handler(event, context):
    task_token = event["task_token"]
    approved = event["approved"]  # True ou False

    if approved:
        sfn.send_task_success(
            taskToken=task_token,
            output=json.dumps({"approved": True}),
        )
    else:
        sfn.send_task_success(
            taskToken=task_token,
            output=json.dumps({"approved": False}),
        )
```

L'approbation peut être déclenchée via :
- Un **lien dans le message Teams** (API Gateway → Lambda)
- L'**interface AWS Step Functions** dans la console
- Un **bouton dans l'UI Streamlit** qui appelle l'API d'approbation

---

## 9. Amazon Bedrock — LLM juge pour RAGAS

Remplace l'appel direct à l'API Mistral.

### 9.1. Mistral sur Bedrock

Amazon Bedrock propose les modèles Mistral en serverless (pas de provisionnement d'instance) :

| Modèle | Model ID Bedrock | Usage | Coût approximatif |
|---|---|---|---|
| Mistral Large | `mistral.mistral-large-latest` | LLM juge haute qualité | ~8$/M input tokens |
| Mistral Small | `mistral.mistral-small-latest` | LLM juge économique | ~1$/M input tokens |
| Mistral 7B Instruct | `mistral.mistral-7b-instruct-v0:2` | Inférence rapide | ~0.15$/M input tokens |

### 9.2. Intégration avec RAGAS via LangChain

```python
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = LangchainLLMWrapper(
    ChatBedrock(
        model_id="mistral.mistral-small-latest",
        region_name="eu-west-1",
        model_kwargs={"temperature": 0},
    )
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="eu-west-1",
    )
)

result = evaluate(
    dataset=ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)
```

### 9.3. Comparaison Mistral API directe vs Bedrock

| Critère | Mistral API (actuel) | Amazon Bedrock |
|---|---|---|
| Authentification | Clé API (`MISTRAL_API_KEY`) | IAM Role (pas de clé en clair) |
| Latence réseau | Internet (variable) | Réseau AWS interne (~5ms) |
| Conformité données | Données transitent par Mistral | Données restent dans AWS |
| Disponibilité | Dépend de l'API Mistral | SLA AWS 99.9% |
| Facturation | Compte Mistral séparé | Facture AWS consolidée |
| Modèles disponibles | Tous les modèles Mistral | Mistral + Claude + Llama + Titan + ... |

---

## 10. Amazon SNS + AWS Chatbot — Notifications Teams

Remplace les notifications Teams directes (webhook).

### 10.1. Architecture de notification

```
Step Functions ──► Amazon SNS ──► AWS Chatbot ──► Microsoft Teams
                   (topic)        (channel)       (webhook)

EventBridge  ──► Amazon SNS ──► AWS Chatbot ──► Microsoft Teams
  (rules)        (même topic)
```

### 10.2. Configuration SNS

```python
sns = boto3.client("sns")

topic = sns.create_topic(
    Name="mlops-notifications",
    Tags=[{"Key": "project", "Value": "mlops"}],
)
```

### 10.3. AWS Chatbot pour Teams

AWS Chatbot supporte nativement Microsoft Teams depuis 2023. Configuration :

1. Créer un **AWS Chatbot client** pour Microsoft Teams dans la console AWS
2. Autoriser le bot dans le tenant Microsoft 365
3. Associer le topic SNS au channel Teams

```python
chatbot = boto3.client("chatbot")

chatbot.create_microsoft_teams_channel_configuration(
    ConfigurationName="mlops-teams-channel",
    ChannelId="<teams-channel-id>",
    TeamId="<teams-team-id>",
    TenantId="<microsoft-tenant-id>",
    SnsTopicArns=[topic_arn],
    IamRoleArn=chatbot_role_arn,
    LoggingLevel="INFO",
)
```

### 10.4. Messages personnalisés

Pour des messages riches (Adaptive Cards), combiner SNS avec une Lambda de formatage :

```python
def format_teams_notification(event, context):
    message = json.loads(event["Records"][0]["Sns"]["Message"])
    run_id = message["run_id"]
    ml_score = message.get("ml_score", "N/A")
    model_name = message.get("model_name", "inconnu")
    event_type = message.get("event_type", "")

    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {"type": "TextBlock", "size": "Large", "weight": "Bolder",
                     "text": f"MLOps — {model_name}"},
                    {"type": "TextBlock", "text": _message_for_event(event_type, ml_score), "wrap": True},
                    {"type": "FactSet", "facts": [
                        {"title": "Run ID", "value": str(run_id)},
                        {"title": "MLScore", "value": str(ml_score)},
                    ]},
                ],
                "actions": [
                    {"type": "Action.OpenUrl", "title": "Ouvrir SageMaker Studio",
                     "url": f"https://studio.{REGION}.sagemaker.aws"},
                    {"type": "Action.OpenUrl", "title": "Approuver / Rejeter",
                     "url": f"{API_URL}/approve?run_id={run_id}"},
                ],
            },
        }],
    }

    requests.post(TEAMS_WEBHOOK_URL, json=card)
```

---

## 11. Amazon RDS for PostgreSQL

Remplace le container PostgreSQL 16.

### 11.1. Configuration

| Propriété | Valeur |
|---|---|
| Engine | PostgreSQL 16 |
| Instance | db.t4g.micro (2 vCPU, 1 GiB RAM) |
| Stockage | 20 GiB gp3 (auto-scaling jusqu'à 100 GiB) |
| Multi-AZ | Oui (production) / Non (dev) |
| Backup | Automatique, rétention 7 jours |
| Encryption | AES-256 (AWS KMS) |
| Networking | Subnets privés uniquement |
| Identifier | `mlops-db-production` |

### 11.2. Base de données

| Base | Usage |
|---|---|
| `mlops` | Application (datasets, pipeline_runs, run_results) |

> Note : pas besoin de la base `mlflow` — SageMaker gère son propre tracking.

### 11.3. Chaîne de connexion

Stockée dans AWS Secrets Manager, rotation automatique :

```json
{
  "engine": "postgres",
  "host": "mlops-db-production.xxxxx.eu-west-1.rds.amazonaws.com",
  "port": 5432,
  "dbname": "mlops",
  "username": "mlopsadmin",
  "password": "auto-rotated-secret"
}
```

Accès depuis l'application :

```python
import boto3
import json

def get_database_url():
    client = boto3.client("secretsmanager")
    secret = json.loads(
        client.get_secret_value(SecretId="mlops/database")["SecretString"]
    )
    return (
        f"postgresql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )
```

---

## 12. Amazon S3 — Données et artefacts

Remplace les volumes Docker (`model_outputs`, `mlflow_artifacts`, `./data`).

### 12.1. Buckets

| Bucket | Contenu | Lifecycle |
|---|---|---|
| `mlops-datasets-{account}` | Fichiers JSONL d'entraînement et d'évaluation | Permanent |
| `mlops-model-outputs-{account}` | Modèles entraînés (poids, tokenizer, config) | Glacier après 90 jours |
| `mlops-sagemaker-{account}` | Artefacts SageMaker (logs, checkpoints, métriques) | Géré par SageMaker |

### 12.2. Structure des objets

```
mlops-datasets-{account}/
├── train/
│   ├── rag_qa_train.jsonl
│   ├── medical_qa_train.jsonl
│   └── legal_qa_train.jsonl
└── eval/
    ├── ragas_eval.jsonl
    ├── medical_ragas_eval.jsonl
    └── legal_ragas_eval.jsonl

mlops-model-outputs-{account}/
└── mistral-7b-rag-qa/
    └── output/
        ├── model.tar.gz          # Archive SageMaker standard
        └── inference_results.json
```

### 12.3. Versioning et lifecycle

```python
s3 = boto3.client("s3")

s3.put_bucket_versioning(
    Bucket="mlops-model-outputs-xxxx",
    VersioningConfiguration={"Status": "Enabled"},
)

s3.put_bucket_lifecycle_configuration(
    Bucket="mlops-model-outputs-xxxx",
    LifecycleConfiguration={
        "Rules": [{
            "ID": "archive-old-models",
            "Status": "Enabled",
            "Transitions": [
                {"Days": 90, "StorageClass": "GLACIER"},
            ],
            "NoncurrentVersionTransitions": [
                {"NoncurrentDays": 30, "StorageClass": "GLACIER"},
            ],
        }],
    },
)
```

---

## 13. AWS Secrets Manager — Secrets

Remplace le fichier `.env`.

### 13.1. Secrets stockés

| Secret ID | Contenu | Rotation |
|---|---|---|
| `mlops/database` | Host, port, username, password PostgreSQL | Auto (30 jours) |
| `mlops/hf-token` | Token HuggingFace Hub | Manuelle |
| `mlops/teams-webhook` | URL du webhook Teams | Manuelle |

### 13.2. Pas de clé API Bedrock

Contrairement à l'API Mistral directe, Amazon Bedrock utilise l'**IAM Role** du service appelant. Aucune clé API n'est nécessaire — c'est l'un des avantages majeurs de l'approche AWS-native.

### 13.3. Rotation automatique (RDS)

```python
sm = boto3.client("secretsmanager")

sm.rotate_secret(
    SecretId="mlops/database",
    RotationLambdaARN=rotation_lambda_arn,
    RotationRules={"AutomaticallyAfterDays": 30},
)
```

---

## 14. Amazon CloudWatch — Observabilité

Remplace `docker compose logs` et les métriques manuelles.

### 14.1. Sources de logs

| Source | Log Group | Rétention |
|---|---|---|
| App Runner (API) | `/aws/apprunner/mlops-api` | 30 jours |
| App Runner (UI) | `/aws/apprunner/mlops-ui` | 30 jours |
| Lambda (Eval) | `/aws/lambda/mlops-eval` | 14 jours |
| Lambda (Dispatcher) | `/aws/lambda/mlops-training-dispatcher` | 14 jours |
| SageMaker Training | `/aws/sagemaker/TrainingJobs` | 30 jours |
| Step Functions | `/aws/stepfunctions/mlops-release` | 30 jours |
| EventBridge | (via CloudTrail) | 90 jours |

### 14.2. Métriques custom

L'API et les workers publient des métriques custom dans CloudWatch :

```python
cloudwatch = boto3.client("cloudwatch")

cloudwatch.put_metric_data(
    Namespace="MLOps",
    MetricData=[
        {
            "MetricName": "MLScore",
            "Value": 0.82,
            "Unit": "None",
            "Dimensions": [
                {"Name": "ModelName", "Value": "mistral-7b-rag-qa"},
                {"Name": "RunId", "Value": "42"},
            ],
        },
        {
            "MetricName": "TrainingDuration",
            "Value": 2700,
            "Unit": "Seconds",
            "Dimensions": [
                {"Name": "ModelName", "Value": "mistral-7b-rag-qa"},
            ],
        },
    ],
)
```

### 14.3. Alertes

| Alerte | Condition | Sévérité | Action |
|---|---|---|---|
| Training job échoué | SageMaker TrainingJobStatus = Failed | ALARM | SNS → Teams |
| Lambda eval timeout | Lambda Duration > 800000ms | ALARM | SNS → Teams |
| API erreurs 5xx | App Runner 5xxCount > 5 en 5 min | ALARM | SNS → Teams |
| MLScore dégradé | Custom MLOps/MLScore < 0.5 (p50 sur 5 runs) | WARNING | SNS → Email |
| SQS DLQ non vide | ApproximateNumberOfMessagesVisible > 0 | WARNING | SNS → Email |
| Step Functions échec | ExecutionsFailed > 0 | ALARM | SNS → Teams |

### 14.4. Dashboard CloudWatch

```
┌──────────────────────────────────────────────────────────────┐
│                    MLOps Operations Dashboard                 │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Runs actifs    │  │  MLScore moyen  │  │ Taux succès  │ │
│  │      3          │  │     0.742       │  │    87%       │ │
│  │ (Step Functions │  │ (Custom Metric) │  │ (SageMaker)  │ │
│  │  running)       │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  MLScore par run (CloudWatch Custom Metrics)            │ │
│  │  ████████████ 0.81  mistral-7b-rag-qa                   │ │
│  │  █████████ 0.67     phi-2-medical                       │ │
│  │  ███████████ 0.74   llama-2-legal                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │  SageMaker Training     │  │  Lambda Eval Duration     │ │
│  │  Duration (p50)         │  │  (p95)                    │ │
│  │  45 min                 │  │  8.2 min                  │ │
│  └──────────────────────────┘  └──────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SQS DLQ Messages (mlops-training-dlq)                  │ │
│  │  0 ✓                                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 14.5. AWS X-Ray (Tracing distribué)

Activer X-Ray sur l'API et les Lambda pour tracer les requêtes de bout en bout :

```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.httplib import add_ignored as httplib_add_ignored

xray_recorder.configure(service="mlops-api")
```

---

## 15. Sécurité et identité (IAM)

### 15.1. IAM Roles

| Service | IAM Role | Permissions |
|---|---|---|
| App Runner (API) | `mlops-api-role` | SecretsManager:GetSecretValue, Events:PutEvents, S3:GetObject, RDS connect |
| App Runner (UI) | `mlops-ui-role` | (aucune — appelle l'API uniquement via HTTPS) |
| Lambda (Eval) | `mlops-eval-role` | SQS:ReceiveMessage, Bedrock:InvokeModel, S3:GetObject, S3:PutObject, Events:PutEvents, SecretsManager:GetSecretValue |
| Lambda (Dispatcher) | `mlops-dispatcher-role` | SQS:ReceiveMessage, SageMaker:CreateTrainingJob |
| Lambda (Approval) | `mlops-approval-role` | StepFunctions:SendTaskSuccess, StepFunctions:SendTaskFailure |
| SageMaker Training | `mlops-sagemaker-role` | S3:GetObject, S3:PutObject, ECR:GetDownloadUrlForLayer, CloudWatch:PutMetricData, SecretsManager:GetSecretValue |
| Step Functions | `mlops-stepfunctions-role` | SageMaker:*, Lambda:InvokeFunction, SNS:Publish, Events:PutEvents |
| AWS Chatbot | `mlops-chatbot-role` | SNS:Subscribe, CloudWatch:DescribeAlarms |

### 15.2. Politique de moindre privilège — Exemple API

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadSecrets",
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": ["arn:aws:secretsmanager:eu-west-1:*:secret:mlops/*"]
    },
    {
      "Sid": "PublishEvents",
      "Effect": "Allow",
      "Action": ["events:PutEvents"],
      "Resource": ["arn:aws:events:eu-west-1:*:event-bus/mlops-events"]
    },
    {
      "Sid": "ReadDatasets",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::mlops-datasets-*",
        "arn:aws:s3:::mlops-datasets-*/*"
      ]
    }
  ]
}
```

### 15.3. Réseau (VPC)

```
┌──────────────────────────────────────────────┐
│          VPC: vpc-mlops (10.0.0.0/16)        │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Public Subnet A (10.0.1.0/24) — AZ-a   │ │
│  │ → NAT Gateway                           │ │
│  │ → App Runner VPC Connector              │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Public Subnet B (10.0.2.0/24) — AZ-b   │ │
│  │ → NAT Gateway (redundancy)             │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Private Subnet A (10.0.10.0/24) — AZ-a │ │
│  │ → RDS Primary                           │ │
│  │ → Lambda (VPC)                          │ │
│  │ → SageMaker Compute                     │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Private Subnet B (10.0.11.0/24) — AZ-b │ │
│  │ → RDS Standby (Multi-AZ)               │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  VPC Endpoints (PrivateLink):                │
│  → S3 (Gateway)                              │
│  → Secrets Manager (Interface)               │
│  → SageMaker API (Interface)                 │
│  → EventBridge (Interface)                   │
│  → ECR (Interface)                           │
│  → Bedrock Runtime (Interface)               │
└──────────────────────────────────────────────┘
```

---

## 16. Infrastructure as Code (AWS CDK)

### 16.1. Structure du projet CDK

```
infra/
├── bin/
│   └── mlops-infra.ts            # Point d'entrée CDK
├── lib/
│   ├── mlops-stack.ts            # Stack principale (orchestration)
│   ├── networking-stack.ts       # VPC, subnets, NAT, endpoints
│   ├── database-stack.ts         # RDS PostgreSQL
│   ├── storage-stack.ts          # S3 buckets
│   ├── secrets-stack.ts          # Secrets Manager
│   ├── eventbridge-stack.ts      # Event bus, rules
│   ├── sqs-stack.ts              # Queues + DLQ
│   ├── ecr-stack.ts              # Container Registry
│   ├── apprunner-stack.ts        # API + UI
│   ├── sagemaker-stack.ts        # ML Workspace, Compute
│   ├── lambda-stack.ts           # Eval, Dispatcher, Approval
│   ├── stepfunctions-stack.ts    # Release workflow
│   ├── notifications-stack.ts   # SNS + Chatbot
│   └── monitoring-stack.ts       # CloudWatch, Alarms, Dashboard
├── cdk.json
├── tsconfig.json
└── package.json
```

### 16.2. Extrait `stepfunctions-stack.ts`

```typescript
import * as cdk from 'aws-cdk-lib';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';

export class StepFunctionsStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props: StepFunctionsStackProps) {
    super(scope, id, props);

    const startTraining = new tasks.SageMakerCreateTrainingJob(this, 'StartTraining', {
      trainingJobName: sfn.JsonPath.format('mlops-train-{}-{}',
        sfn.JsonPath.stringAt('$.detail.run_id'),
        sfn.JsonPath.stringAt('$$.Execution.Name')
      ),
      algorithmSpecification: {
        trainingImage: tasks.DockerImage.fromEcrRepository(props.trainingRepo),
        trainingInputMode: tasks.InputMode.FILE,
      },
      inputDataConfig: [{
        channelName: 'train',
        dataSource: { s3DataSource: { s3Uri: 's3://mlops-datasets/train/' } },
      }],
      outputDataConfig: { s3OutputPath: 's3://mlops-model-outputs/' },
      resourceConfig: {
        instanceType: new ec2.InstanceType('ml.g4dn.xlarge'),
        instanceCount: 1,
        volumeSize: cdk.Size.gibibytes(50),
      },
      integrationPattern: sfn.IntegrationPattern.RUN_JOB,
    });

    const checkMLScore = new sfn.Choice(this, 'CheckMLScore')
      .when(sfn.Condition.booleanEquals('$.evalResult.passed', true),
        notifyApproval)
      .otherwise(autoReject);

    const definition = notifyLoading
      .next(registerModel)
      .next(checkTaskType)
      // ... rest of the state machine

    const stateMachine = new sfn.StateMachine(this, 'ReleaseWorkflow', {
      definitionBody: sfn.DefinitionBody.fromChainable(definition),
      timeout: cdk.Duration.days(7),
      tracingEnabled: true,
    });

    new events.Rule(this, 'RunCreatedRule', {
      eventBus: props.eventBus,
      eventPattern: { detailType: ['run.created'] },
      targets: [new targets.SfnStateMachine(stateMachine)],
    });
  }
}
```

### 16.3. Déploiement

```bash
# Bootstrap (une seule fois par compte/région)
cdk bootstrap aws://ACCOUNT/eu-west-1

# Synthétiser le template CloudFormation
cdk synth

# Déployer
cdk deploy --all --require-approval broadening

# Détruire (attention: destructif)
cdk destroy --all
```

---

## 17. Workflow de bout en bout

### Séquence complète

```
┌──────────┐  HTTPS   ┌──────────────┐ PutEvents  ┌───────────────────┐
│Streamlit │────────►│   FastAPI    │──────────►│  EventBridge      │
│App Runner│         │  App Runner  │            │  Bus: mlops-events│
└──────────┘         └──────┬───────┘            └────────┬──────────┘
                            │                             │
                      ┌─────┴─────┐         ┌─────────────┼──────────────┐
                      │           │         │             │              │
                      ▼           │         ▼             ▼              ▼
              ┌──────────────┐    │  ┌────────────┐ ┌──────────┐  ┌──────────┐
              │ Amazon RDS   │    │  │   SQS      │ │  Step    │  │   SNS    │
              │ PostgreSQL   │    │  │  Queues    │ │Functions │  │  Topic   │
              └──────────────┘    │  └─────┬──────┘ │ (Release │  └────┬─────┘
                                  │        │        │ Workflow) │       │
                                  │        ▼        └────┬─────┘       ▼
                                  │  ┌──────────┐       │        ┌──────────┐
                                  │  │  Lambda  │       │        │   AWS    │
                                  │  │ Dispatch │       │        │ Chatbot  │
                                  │  └────┬─────┘       │        │ → Teams  │
                                  │       │              │        └──────────┘
                                  │       ▼              │
                                  │  ┌──────────────┐    │
                                  │  │  SageMaker   │    │
                                  │  │  Training    │◄───┘ (CreateTrainingJob.sync)
                                  │  │  Job (GPU)   │
                                  │  └──────┬───────┘
                                  │         │
                                  │         ▼
                                  │  ┌──────────────┐
                                  │  │   Lambda /   │
                                  │  │ ECS Fargate  │
                                  │  │  (Eval +     │
                                  │  │   RAGAS)     │
                                  │  └──────┬───────┘
                                  │         │
                                  │         ▼
                                  │  ┌──────────────┐
                                  │  │   Amazon     │
                                  │  │   Bedrock    │
                                  │  │  (Mistral)   │
                                  │  └──────────────┘
                                  │
                                  ▼
                           ┌─────────────────────────────────┐
                           │        Amazon SageMaker          │
                           │                                  │
                           │  ┌────────────┐  ┌────────────┐ │
                           │  │Experiments │  │  Model     │ │
                           │  │(tracking)  │  │  Registry  │ │
                           │  └────────────┘  └────────────┘ │
                           └─────────────────────────────────┘
```

### Tableau récapitulatif

| # | Action | Service AWS | Détail |
|---|---|---|---|
| 1 | User clique "Lancer" | App Runner (Streamlit) | POST vers l'API |
| 2 | L'API crée le run | App Runner (FastAPI) + RDS | Status = `pending` |
| 3 | L'API publie `run.created` | EventBridge | Bus `mlops-events` |
| 4 | Règle EventBridge → Step Functions | Step Functions | Démarre le workflow de release |
| 5 | State `RegisterBaseModel` | SageMaker Model Registry | `PendingManualApproval` |
| 6 | State `StartTraining` (.sync) | SageMaker Training Job | GPU ml.g4dn.xlarge (Spot) |
| 7 | Training terminé | EventBridge | Publie `run.training_done` |
| 8 | State `StartEvaluation` (TaskToken) | Lambda / ECS Fargate | RAGAS + Bedrock (Mistral) |
| 9 | State `CheckMLScore` | Step Functions Choice | ≥ 0.7 ou < 0.7 |
| 10a | MLScore ≥ seuil | Step Functions | → `WaitForHumanApproval` |
| 10b | MLScore < seuil | Step Functions | → `AutoReject` + tag `rejected` |
| 11 | Notification Teams | SNS → Chatbot → Teams | Adaptive Card |
| 12 | Gate manuelle (TaskToken, 72h) | Step Functions + Lambda | Callback pattern |
| 13a | Approuvé | SageMaker | `Approved` + tag `go_prod` |
| 13b | Rejeté | SageMaker | `Rejected` + tag `rejected` |

---

## 18. Estimation des coûts

### Coût mensuel estimé (usage modéré : ~10 runs/mois)

| Service | Configuration | Estimation mensuelle |
|---|---|---|
| Amazon RDS PostgreSQL | db.t4g.micro, 20 GiB, Single-AZ | ~15 € |
| AWS App Runner (API + UI) | 2 services, 0.25–0.5 vCPU | ~10 € |
| Amazon EventBridge | ~100 événements/mois | ~0.01 € |
| Amazon SQS | ~200 messages/mois | ~0.01 € |
| AWS Lambda (Eval + Dispatcher + Approval) | ~50 invocations, 1 GB, 15 min max | ~2 € |
| SageMaker Training Jobs | ml.g4dn.xlarge, 10 runs × 30 min, Spot | ~20 € (avec Spot: ~6 €) |
| SageMaker (Experiments + Registry) | — | Gratuit (coût = stockage S3) |
| Amazon Bedrock (Mistral Small) | ~500K tokens/run × 10 runs | ~5 € |
| Amazon S3 | ~50 GiB Standard | ~1 € |
| AWS Secrets Manager | 3 secrets | ~1.20 € |
| Amazon SNS + Chatbot | ~30 notifications | ~0.01 € |
| AWS Step Functions | ~10 exécutions, ~200 transitions | ~0.01 € |
| Amazon ECR | 3 images, ~5 GiB | ~0.50 € |
| CloudWatch | Logs (10 GiB), 5 alertes, dashboard | ~5 € |
| **TOTAL (avec Spot)** | | **~46 €/mois** |
| **TOTAL (sans Spot)** | | **~60 €/mois** |

### Comparaison des trois architectures

| Aspect | Docker Compose | Azure Native | AWS Native |
|---|---|---|---|
| Coût mensuel (~10 runs) | 200–500 € (machine GPU) | ~116 € | ~46–60 € |
| GPU idle | Payé 24/7 | Scale-to-zero | Scale-to-zero + **Spot** (90% éco.) |
| Serverless | Non | Partiel (Container Apps) | Maximal (Lambda, EventBridge, Step Functions) |
| Maintenance | OS, Docker, sécurité | PaaS managé | Serverless (zéro serveur) |
| Gate manuelle | Non existante | Azure DevOps Environment | Step Functions Callback |
| Notifications | Non existantes | Logic Apps → Teams | SNS + Chatbot → Teams |
| Orchestration | Polling HTTP | Azure DevOps Pipeline (YAML) | Step Functions (ASL JSON, visuel) |
| IaC | docker-compose.yml | Bicep | AWS CDK (TypeScript) |
| LLM juge | Mistral API directe | Azure AI Services | Amazon Bedrock |
| Replay d'événements | Non | Non | EventBridge Archive (90 jours) |

---

## 19. Plan de migration

### Phase 1 — Fondations (Semaine 1-2)

1. Configurer le compte AWS (Organizations, billing alerts)
2. Déployer le VPC avec CDK (subnets, NAT, VPC endpoints)
3. Déployer RDS PostgreSQL + migrer le schéma
4. Déployer S3 + uploader les datasets
5. Déployer Secrets Manager + créer les secrets
6. Déployer ECR + build/push les images Docker

### Phase 2 — API et UI (Semaine 2-3)

7. Déployer App Runner API (FastAPI)
8. Adapter l'API pour EventBridge (`publish_event()`)
9. Déployer App Runner UI (Streamlit)
10. Configurer les VPC Connectors (accès RDS)
11. Tester la chaîne UI → API → RDS

### Phase 3 — Événements et queues (Semaine 3)

12. Créer le bus EventBridge `mlops-events`
13. Créer les queues SQS + DLQ
14. Créer les règles EventBridge (routing)
15. Tester le fan-out (un événement → plusieurs cibles)

### Phase 4 — Training sur SageMaker (Semaine 3-4)

16. Créer le Compute Cluster SageMaker (ou utiliser des jobs éphémères)
17. Adapter le training worker pour SageMaker (packaging, S3 I/O)
18. Déployer la Lambda dispatcher
19. Configurer le MLflow tracking vers SageMaker (ou SageMaker Experiments)
20. Tester un training de bout en bout

### Phase 5 — Évaluation sur Bedrock (Semaine 4-5)

21. Activer Mistral Small/Large sur Amazon Bedrock
22. Adapter l'evaluation worker (LangChain Bedrock)
23. Déployer la Lambda d'évaluation (ou ECS Fargate Task)
24. Tester l'évaluation RAGAS avec Bedrock

### Phase 6 — Orchestration Step Functions (Semaine 5-6)

25. Déployer la machine à états Step Functions
26. Implémenter la Lambda d'approbation (callback pattern)
27. Configurer SNS + AWS Chatbot pour Teams
28. Connecter EventBridge → Step Functions (règle `run.created`)
29. Tester le workflow complet : UI → Training → Eval → Approval → Tag

### Phase 7 — Monitoring et Hardening (Semaine 6-7)

30. Configurer CloudWatch Logs, Metrics, Alarms
31. Activer X-Ray sur l'API et les Lambda
32. Créer le dashboard opérationnel
33. Configurer les VPC Endpoints (PrivateLink)
34. Tests de charge et revue de sécurité IAM
