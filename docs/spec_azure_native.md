# Spécification — Équivalent Azure-Native du projet MLOps

> Transposition complète de l'architecture MLOps (FastAPI, Streamlit, MLflow, PostgreSQL, workers, RabbitMQ, Digital.ai Release) vers une stack 100% Azure.

---

## Table des matières

1. [Matrice de correspondance](#1-matrice-de-correspondance)
2. [Architecture Azure cible](#2-architecture-azure-cible)
3. [Azure Machine Learning Workspace](#3-azure-machine-learning-workspace)
4. [Azure Container Apps — API et UI](#4-azure-container-apps--api-et-ui)
5. [Azure Service Bus — Messagerie événementielle](#5-azure-service-bus--messagerie-événementielle)
6. [Azure ML Compute — Training](#6-azure-ml-compute--training)
7. [Azure Container Apps Jobs — Evaluation](#7-azure-container-apps-jobs--evaluation)
8. [Azure DevOps Pipelines — Orchestration de release](#8-azure-devops-pipelines--orchestration-de-release)
9. [Azure Logic Apps — Notifications Teams](#9-azure-logic-apps--notifications-teams)
10. [Azure Database for PostgreSQL](#10-azure-database-for-postgresql)
11. [Azure Blob Storage — Données et artefacts](#11-azure-blob-storage--données-et-artefacts)
12. [Azure Key Vault — Secrets](#12-azure-key-vault--secrets)
13. [Azure Monitor — Observabilité](#13-azure-monitor--observabilité)
14. [Sécurité et identité](#14-sécurité-et-identité)
15. [Infrastructure as Code (Bicep)](#15-infrastructure-as-code-bicep)
16. [Workflow de bout en bout](#16-workflow-de-bout-en-bout)
17. [Estimation des coûts](#17-estimation-des-coûts)
18. [Plan de migration](#18-plan-de-migration)

---

## 1. Matrice de correspondance

| Composant actuel | Service Azure | SKU recommandé | Justification |
|---|---|---|---|
| **PostgreSQL 16** (Docker) | Azure Database for PostgreSQL — Flexible Server | Burstable B1ms | PaaS managé, backup auto, haute dispo |
| **FastAPI** (Docker) | Azure Container Apps | Consumption (0.25 vCPU / 0.5 Gi) | Serverless, scale-to-zero, HTTP ingress natif |
| **Streamlit UI** (Docker) | Azure Container Apps | Consumption (0.25 vCPU / 0.5 Gi) | Même environnement que l'API, ingress HTTPS |
| **MLflow Tracking** (Docker) | Azure Machine Learning Workspace | Standard | Tracking natif, Model Registry, Endpoints |
| **MLflow Model Registry** | Azure ML Model Registry | (inclus dans Workspace) | Versionning, tags, stages (Production/Staging/Archived) |
| **Training Worker** (Docker, poll) | Azure ML Compute Cluster | Standard_NC6s_v3 (GPU) ou Standard_DS3_v2 (CPU) | Auto-scale 0→N nœuds, facturation à l'usage |
| **Evaluation Worker** (Docker, poll) | Azure Container Apps Job | Consumption (1 vCPU / 2 Gi) | Exécution événementielle via Service Bus trigger |
| **RabbitMQ** (Docker) | Azure Service Bus | Standard | Files, topics, sessions, dead-letter natifs |
| **Digital.ai Release** | Azure DevOps Pipelines | Basic (gratuit 1 parallel job) | Stages, environments, approval gates, audit trail |
| **Teams notifications** | Azure Logic Apps | Consumption | Connecteur Teams natif, Adaptive Cards |
| **Mistral API** (LLM juge RAGAS) | Azure AI Services — Models as a Service | Mistral Large / Small sur Azure AI | Déploiement serverless, même API, données dans Azure |
| **HuggingFace Hub** | Azure ML Model Catalog + HuggingFace on Azure | — | Modèles pré-intégrés, déploiement one-click |
| **Volumes Docker** | Azure Blob Storage + Azure Files | Standard LRS | Stockage persistant des modèles et datasets |
| **`.env` / Secrets** | Azure Key Vault | Standard | Rotation auto, audit, Managed Identity |
| **Monitoring** | Azure Monitor + Application Insights | — | Métriques, logs, alertes, dashboards |
| **Container Registry** | Azure Container Registry | Basic | Stockage des images Docker |
| **IaC** | Bicep / Terraform | — | Déploiement reproductible |

---

## 2. Architecture Azure cible

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              AZURE RESOURCE GROUP                                │
│                              rg-mlops-production                                 │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                    AZURE CONTAINER APPS ENVIRONMENT                        │ │
│  │                                                                             │ │
│  │   ┌──────────────┐     ┌──────────────┐                                    │ │
│  │   │  Streamlit   │     │   FastAPI    │                                    │ │
│  │   │  Container   │────►│  Container   │                                    │ │
│  │   │    App       │     │    App       │                                    │ │
│  │   │  (HTTPS)     │     │  (HTTPS)     │                                    │ │
│  │   └──────────────┘     └──────┬───────┘                                    │ │
│  │                               │                                             │ │
│  │                               │ publish                                     │ │
│  │                               ▼                                             │ │
│  │                    ┌─────────────────────┐                                  │ │
│  │                    │  Azure Service Bus  │                                  │ │
│  │                    │  (Topics & Subs)    │                                  │ │
│  │                    └─────────┬───────────┘                                  │ │
│  │                              │                                              │ │
│  │               ┌──────────────┼──────────────┐                               │ │
│  │               ▼              ▼              ▼                               │ │
│  │    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                      │ │
│  │    │  Container   │ │  Container   │ │   Azure ML   │                      │ │
│  │    │  Apps Job    │ │  Apps Job    │ │   Compute    │                      │ │
│  │    │  (Eval)      │ │  (Release)   │ │  (Training)  │                      │ │
│  │    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                      │ │
│  │           │                │                 │                               │ │
│  └───────────┼────────────────┼─────────────────┼───────────────────────────────┘ │
│              │                │                 │                                  │
│              ▼                ▼                 ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                              │ │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │   │  Azure ML    │  │   Azure DB   │  │ Azure Blob   │  │  Azure Key   │   │ │
│  │   │  Workspace   │  │  PostgreSQL  │  │   Storage    │  │    Vault     │   │ │
│  │   │  (Tracking   │  │  Flex Server │  │  (datasets   │  │  (secrets)   │   │ │
│  │   │   + Registry)│  │              │  │   + modèles) │  │              │   │ │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  │                                                                              │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │  Azure DevOps        │  │  Azure Logic Apps    │  │  Azure AI Services  │   │
│  │  Pipelines           │  │  (→ Teams)           │  │  (Mistral on Azure) │   │
│  │  (Release workflow)  │  │                      │  │                      │   │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │  Azure Monitor  +  Application Insights  +  Log Analytics Workspace         │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Azure Machine Learning Workspace

Remplace MLflow Tracking Server + MLflow Model Registry.

### 3.1. Tracking des expériences

| Fonctionnalité MLflow | Équivalent Azure ML |
|---|---|
| `mlflow.set_experiment()` | `azure.ai.ml.MLClient.create_or_update(experiment)` ou SDK MLflow compatible |
| `mlflow.start_run()` | Identique (Azure ML expose un endpoint MLflow compatible) |
| `mlflow.log_metrics()` | Identique via l'endpoint MLflow |
| `mlflow.log_artifacts()` | Identique, stocké dans Azure Blob Storage |
| `mlflow.set_tag()` | Identique |
| MLflow UI (`http://mlflow:5000`) | Azure ML Studio (`https://ml.azure.com`) |

Azure ML Workspace expose nativement un **MLflow Tracking URI** compatible. Les workers existants peuvent continuer à utiliser le SDK MLflow sans modification majeure :

```python
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# → "azureml://westeurope.api.azureml.ms/mlflow/v1.0/subscriptions/{sub}/
#    resourceGroups/{rg}/providers/Microsoft.MachineLearningServices/workspaces/{ws}"

mlflow.set_experiment("rag-qa-finetune")
with mlflow.start_run(run_name="mistral-7b-rag-qa"):
    mlflow.log_metrics({"train_loss": 0.42, "perplexity": 1.52})
    mlflow.log_artifacts("./model", "model")
```

### 3.2. Model Registry

| Fonctionnalité MLflow | Équivalent Azure ML |
|---|---|
| `create_registered_model()` | `ml_client.models.create_or_update(Model(...))` |
| `create_model_version()` | Versioning automatique à chaque `create_or_update` |
| `set_model_version_tag()` | `model.tags = {"lifecycle": "finetuned", "validation": "validated"}` |
| Stage `Production` / `Archived` | Tags custom `deployment_status: go_prod / rejected` |
| Model URI `runs:/{id}/model` | `azureml://registries/{registry}/models/{name}/versions/{v}` |

Exemple d'enregistrement d'un modèle :

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="rg-mlops-production",
    workspace_name="mlops-workspace",
)

model = Model(
    name="mistral-7b-rag-qa",
    path="azureml://jobs/{job-id}/outputs/model",
    type="custom_model",
    description="Fine-tuned Mistral 7B for RAG QA",
    tags={
        "lifecycle": "finetuned",
        "domain": "medic",
        "validation": "pending",
        "ml_score_threshold": "0.7",
    },
)
registered = ml_client.models.create_or_update(model)
```

### 3.3. Compute Clusters

Remplacent le training worker Docker :

```python
from azure.ai.ml.entities import AmlCompute

gpu_cluster = AmlCompute(
    name="gpu-training-cluster",
    type="amlcompute",
    size="Standard_NC6s_v3",    # 1x V100 GPU, 6 vCPU, 112 GB RAM
    min_instances=0,            # Scale to zero quand inactif
    max_instances=2,
    idle_time_before_scale_down=300,
)
ml_client.compute.begin_create_or_update(gpu_cluster)
```

---

## 4. Azure Container Apps — API et UI

Remplacent les containers Docker FastAPI et Streamlit.

### 4.1. Container App : API FastAPI

| Propriété | Valeur |
|---|---|
| Nom | `ca-mlops-api` |
| Image | `acr-mlops.azurecr.io/mlops-api:latest` |
| CPU / Mémoire | 0.5 vCPU / 1 Gi |
| Ingress | Externe, HTTPS, port 8000 |
| Scale | 1–5 réplicas (basé sur HTTP concurrency) |
| Managed Identity | Activée (accès DB, Key Vault, Service Bus, Azure ML) |

Variables d'environnement (référencées depuis Key Vault) :

```yaml
env:
  - name: DATABASE_URL
    secretRef: database-url
  - name: AZURE_SERVICEBUS_CONNECTION
    secretRef: servicebus-connection
  - name: AZURE_ML_WORKSPACE
    value: mlops-workspace
```

### 4.2. Container App : UI Streamlit

| Propriété | Valeur |
|---|---|
| Nom | `ca-mlops-ui` |
| Image | `acr-mlops.azurecr.io/mlops-ui:latest` |
| CPU / Mémoire | 0.25 vCPU / 0.5 Gi |
| Ingress | Externe, HTTPS, port 8501 |
| Scale | 1–3 réplicas |

```yaml
env:
  - name: API_URL
    value: https://ca-mlops-api.internal.{env}.azurecontainerapps.io
```

### 4.3. Modifications de l'API FastAPI

L'API reste en FastAPI, mais la publication d'événements utilise Azure Service Bus au lieu de RabbitMQ :

```python
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import json
import os

SERVICEBUS_CONN = os.getenv("AZURE_SERVICEBUS_CONNECTION")

def publish_event(topic: str, run_id: int, payload: dict):
    with ServiceBusClient.from_connection_string(SERVICEBUS_CONN) as client:
        sender = client.get_topic_sender(topic_name="mlops-events")
        with sender:
            message = ServiceBusMessage(
                body=json.dumps({
                    "event": topic,
                    "run_id": run_id,
                    "payload": payload,
                }),
                subject=topic,
                application_properties={"event_type": topic},
            )
            sender.send_messages(message)
```

---

## 5. Azure Service Bus — Messagerie événementielle

Remplace RabbitMQ.

### 5.1. Configuration

| Propriété | Valeur |
|---|---|
| Namespace | `sb-mlops-production` |
| SKU | Standard |
| Topic principal | `mlops-events` |
| Région | West Europe |

### 5.2. Topologie des subscriptions

| Subscription | Filtre SQL | Consommateur | Déclencheur |
|---|---|---|---|
| `sub-training` | `event_type = 'run.created' AND task_type = 'finetune'` | Azure ML Job | Lancement du training |
| `sub-evaluation` | `event_type IN ('run.training_done', 'run.created') AND task_type_eval = true` | Container Apps Job (Eval) | Lancement de l'évaluation |
| `sub-release` | `event_type = 'run.created'` | Container Apps Job (Release) | Création de release Azure DevOps |
| `sub-eval-passed` | `event_type = 'run.eval_passed'` | Logic App | Notification Teams + gate |
| `sub-eval-rejected` | `event_type = 'run.eval_rejected'` | Logic App | Notification Teams (rejet) |

### 5.3. Dead-letter queue

Les messages non traités après 3 tentatives sont automatiquement envoyés dans la dead-letter queue du topic, permettant le debug et le re-processing.

### 5.4. Comparaison RabbitMQ vs Service Bus

| Critère | RabbitMQ (Docker) | Azure Service Bus |
|---|---|---|
| Déploiement | Container à gérer | PaaS managé |
| Haute disponibilité | Configuration manuelle | Incluse (Standard SKU) |
| Dead-letter | Plugin à activer | Natif |
| Filtrage des messages | Routing keys | Filtres SQL sur propriétés |
| Taille max message | ~128 Mo | 256 Ko (Standard) / 100 Mo (Premium) |
| Sessions ordonnées | Non natif | Natif |
| Intégration Azure | Aucune | Managed Identity, RBAC, Monitor |
| Coût | Gratuit (self-hosted) | ~10€/mois (Standard) |

---

## 6. Azure ML Compute — Training

Remplace le training worker Docker.

### 6.1. Soumission d'un job de training

Au lieu d'un worker qui poll et exécute le training localement, on soumet un **Azure ML Job** qui s'exécute sur un cluster de calcul managé :

```python
from azure.ai.ml import command, Input
from azure.ai.ml.entities import Environment

training_env = Environment(
    name="mlops-training",
    image="acr-mlops.azurecr.io/mlops-training:latest",
    conda_file="environments/training_conda.yml",
)

training_job = command(
    code="./workers/training",
    command="python worker.py --run-id ${{inputs.run_id}}",
    inputs={
        "run_id": Input(type="string"),
        "train_data": Input(
            type="uri_file",
            path="azureml://datastores/mlops_data/paths/train/rag_qa_train.jsonl",
        ),
    },
    environment=training_env,
    compute="gpu-training-cluster",
    instance_count=1,
    display_name="finetune-mistral-7b",
    experiment_name="rag-qa-finetune",
    tags={"run_id": "42", "model_id": "mistralai/Mistral-7B-v0.1"},
)

returned_job = ml_client.jobs.create_or_update(training_job)
```

### 6.2. Adaptation du training worker

Le worker reste en Python mais s'exécute comme un Azure ML Job plutôt qu'un container Docker permanent :

| Aspect | Docker (actuel) | Azure ML Job (cible) |
|---|---|---|
| Déclenchement | Polling HTTP toutes les 10s | Message Service Bus → soumet un Job |
| Exécution | Container permanent | Job éphémère sur compute cluster |
| GPU | Réservation mémoire 8G | Cluster auto-scale avec GPU V100/A100 |
| Stockage modèle | Volume Docker `model_outputs` | Azure Blob Storage (datastore) |
| Logs | `_log()` → API | Azure ML Run logs (+ API) |
| MLflow | `mlflow.set_tracking_uri("http://mlflow:5000")` | Automatique (workspace intégré) |
| Coût | Container 24/7 | Facturation uniquement pendant l'exécution |

### 6.3. Déclencheur Service Bus → Azure ML Job

Un Container Apps Job léger (le "dispatcher") consomme les messages `run.created` et soumet les jobs Azure ML :

```python
from azure.servicebus import ServiceBusClient
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

def on_message(message):
    event = json.loads(str(message))
    run_id = event["run_id"]
    task_type = event["payload"]["task_type"]

    if task_type == "finetune":
        job = command(
            code="./workers/training",
            command=f"python worker.py --run-id {run_id}",
            compute="gpu-training-cluster",
            environment="mlops-training@latest",
            experiment_name=event["payload"]["experiment_name"],
        )
        ml_client.jobs.create_or_update(job)
```

---

## 7. Azure Container Apps Jobs — Evaluation

Remplace l'evaluation worker Docker.

### 7.1. Pourquoi Container Apps Jobs

L'évaluation RAGAS nécessite des appels API vers Azure AI Services (Mistral), pas de GPU local. Un Container Apps Job déclenché par Service Bus est idéal :

| Propriété | Valeur |
|---|---|
| Nom | `caj-mlops-eval` |
| Type | Event-driven (Service Bus trigger) |
| Image | `acr-mlops.azurecr.io/mlops-eval:latest` |
| CPU / Mémoire | 2 vCPU / 4 Gi |
| Timeout | 3600s (1h) |
| Scale | 0–3 exécutions parallèles |

### 7.2. Configuration du trigger Service Bus

```json
{
  "triggerType": "serviceBus",
  "metadata": {
    "topicName": "mlops-events",
    "subscriptionName": "sub-evaluation",
    "namespace": "sb-mlops-production",
    "messageCount": "1"
  },
  "auth": [{
    "secretRef": "servicebus-connection",
    "triggerParameter": "connection"
  }]
}
```

### 7.3. Adaptation du worker d'évaluation

| Aspect | Docker (actuel) | Container Apps Job (cible) |
|---|---|---|
| Déclenchement | Polling HTTP | Message Service Bus (automatique) |
| LLM juge | `ChatMistralAI` via API Mistral directe | `ChatMistralAI` via Azure AI endpoint |
| Embeddings | `MistralAIEmbeddings` via API Mistral | `MistralAIEmbeddings` via Azure AI endpoint |
| Modèle entraîné | Volume Docker `/app/outputs` | Azure Blob Storage (datastore) |
| Durée de vie | Container permanent | Job éphémère (s'arrête après traitement) |

Configuration Azure AI pour RAGAS :

```python
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

evaluator_llm = ChatMistralAI(
    model="azureai://mistral-small-latest",
    endpoint=os.getenv("AZURE_AI_ENDPOINT"),
    api_key=os.getenv("AZURE_AI_API_KEY"),
    temperature=0,
)
evaluator_embeddings = MistralAIEmbeddings(
    endpoint=os.getenv("AZURE_AI_ENDPOINT"),
    api_key=os.getenv("AZURE_AI_API_KEY"),
)
```

---

## 8. Azure DevOps Pipelines — Orchestration de release

Remplace Digital.ai Release.

### 8.1. Pourquoi Azure DevOps Pipelines

| Critère | Digital.ai Release | Azure DevOps Pipelines |
|---|---|---|
| Intégration Azure | Plugin tiers | Natif (Managed Identity, Service Bus, etc.) |
| Gate d'approbation | Gate native | Environments avec Approvals & Checks |
| Notifications Teams | Plugin custom | Connecteur natif |
| Pipeline as Code | Templates XML | YAML versionné dans le repo |
| Coût | Licence serveur | Gratuit (1 parallel job) puis ~35€/mois |
| Audit trail | Oui | Oui, intégré à Azure DevOps |

### 8.2. Pipeline YAML

```yaml
# azure-pipelines/mlops-release.yml

trigger: none  # Déclenché par le Release Worker via API REST

parameters:
  - name: runId
    type: number
  - name: modelName
    type: string
  - name: modelId
    type: string
  - name: experimentName
    type: string
  - name: taskType
    type: string
    default: finetune

variables:
  - group: mlops-secrets    # Lié à Azure Key Vault
  - name: apiUrl
    value: https://ca-mlops-api.azurecontainerapps.io
  - name: mlscoreThreshold
    value: '0.7'

stages:

  # ═══════════════════════════════════════════════════════════════
  # STAGE 1 : Chargement & Publication du modèle
  # ═══════════════════════════════════════════════════════════════
  - stage: LoadAndRegister
    displayName: 'Chargement & Publication du modèle'
    jobs:
      - job: LoadModel
        displayName: 'Téléchargement et enregistrement MLflow'
        pool:
          vmImage: ubuntu-latest
        steps:
          - task: AzureCLI@2
            displayName: 'Notifier API — status: loading'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              scriptLocation: inlineScript
              inlineScript: |
                curl -X PATCH "${{ variables.apiUrl }}/runs/${{ parameters.runId }}/status?status=loading"

          - task: AzureCLI@2
            displayName: 'Télécharger le modèle HuggingFace'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                pip install huggingface_hub
                python -c "
                from huggingface_hub import snapshot_download
                snapshot_download(
                    '${{ parameters.modelId }}',
                    allow_patterns=['*.json', '*.safetensors', '*.bin', '*.model'],
                    local_dir='$(Build.ArtifactStagingDirectory)/model'
                )
                "

          - task: AzureCLI@2
            displayName: 'Enregistrer dans Azure ML Model Registry'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                az ml model create \
                  --name "${{ parameters.modelName }}" \
                  --path "$(Build.ArtifactStagingDirectory)/model" \
                  --type custom_model \
                  --tags "lifecycle=new" "run_id=${{ parameters.runId }}" \
                  --workspace-name mlops-workspace \
                  --resource-group rg-mlops-production

  # ═══════════════════════════════════════════════════════════════
  # STAGE 2 : Entraînement (Fine-tuning)
  # ═══════════════════════════════════════════════════════════════
  - stage: Training
    displayName: 'Entraînement du modèle'
    dependsOn: LoadAndRegister
    condition: eq('${{ parameters.taskType }}', 'finetune')
    jobs:
      - job: SubmitTrainingJob
        displayName: 'Soumettre le job Azure ML'
        pool:
          vmImage: ubuntu-latest
        steps:
          - task: AzureCLI@2
            displayName: 'Notifier API — status: training'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                curl -X PATCH "${{ variables.apiUrl }}/runs/${{ parameters.runId }}/status?status=training"

          - task: AzureCLI@2
            displayName: 'Soumettre le job de training'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                az ml job create \
                  --file azure-pipelines/jobs/training-job.yml \
                  --set inputs.run_id=${{ parameters.runId }} \
                  --set display_name="train-${{ parameters.modelName }}" \
                  --workspace-name mlops-workspace \
                  --resource-group rg-mlops-production \
                  --stream

  # ═══════════════════════════════════════════════════════════════
  # STAGE 3 : Évaluation RAGAS
  # ═══════════════════════════════════════════════════════════════
  - stage: Evaluation
    displayName: 'Évaluation RAGAS'
    dependsOn: Training
    condition: always()
    jobs:
      - job: RunEvaluation
        displayName: 'Évaluation + Calcul MLScore'
        pool:
          vmImage: ubuntu-latest
        steps:
          - task: AzureCLI@2
            displayName: 'Notifier API — status: evaluating'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                curl -X PATCH "${{ variables.apiUrl }}/runs/${{ parameters.runId }}/status?status=evaluating"

          - task: AzureCLI@2
            displayName: 'Lancer l évaluation'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                az containerapp job start \
                  --name caj-mlops-eval \
                  --resource-group rg-mlops-production \
                  --env-vars "RUN_ID=${{ parameters.runId }}"

          - task: AzureCLI@2
            displayName: 'Attendre et vérifier le MLScore'
            name: checkScore
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                while true; do
                  STATUS=$(curl -s "${{ variables.apiUrl }}/runs/${{ parameters.runId }}" | jq -r '.status')
                  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
                    break
                  fi
                  sleep 30
                done

                MLSCORE=$(curl -s "${{ variables.apiUrl }}/runs/${{ parameters.runId }}" \
                  | jq -r '.results[] | select(.metric_name=="ml_score") | .metric_value')

                echo "MLScore: $MLSCORE"
                echo "##vso[task.setvariable variable=mlScore;isOutput=true]$MLSCORE"

                PASSED=$(python3 -c "print('true' if float('${MLSCORE}') >= float('${{ variables.mlscoreThreshold }}') else 'false')")
                echo "##vso[task.setvariable variable=passed;isOutput=true]$PASSED"

  # ═══════════════════════════════════════════════════════════════
  # STAGE 4a : Rejet automatique (MLScore < seuil)
  # ═══════════════════════════════════════════════════════════════
  - stage: AutoReject
    displayName: 'Rejet automatique'
    dependsOn: Evaluation
    condition: eq(stageDependencies.Evaluation.RunEvaluation.outputs['checkScore.passed'], 'false')
    jobs:
      - job: RejectModel
        steps:
          - task: AzureCLI@2
            displayName: 'Tagger le modèle comme rejeté'
            inputs:
              azureSubscription: 'mlops-service-connection'
              scriptType: bash
              inlineScript: |
                az ml model update \
                  --name "${{ parameters.modelName }}" \
                  --version latest \
                  --set tags.deployment_status=rejected \
                  --set tags.validation=rejected \
                  --workspace-name mlops-workspace \
                  --resource-group rg-mlops-production

                curl -X PATCH "${{ variables.apiUrl }}/runs/${{ parameters.runId }}/status?status=completed" \
                  -d '{"logs": "Modèle rejeté — MLScore insuffisant"}'

  # ═══════════════════════════════════════════════════════════════
  # STAGE 4b : Validation manuelle (MLScore ≥ seuil)
  # ═══════════════════════════════════════════════════════════════
  - stage: ManualValidation
    displayName: 'Validation manuelle'
    dependsOn: Evaluation
    condition: eq(stageDependencies.Evaluation.RunEvaluation.outputs['checkScore.passed'], 'true')
    jobs:
      - deployment: WaitForApproval
        displayName: 'Attente approbation humaine'
        environment: 'mlops-production'    # ← Environment avec approval gate
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureCLI@2
                  displayName: 'Tagger le modèle go_prod'
                  inputs:
                    azureSubscription: 'mlops-service-connection'
                    scriptType: bash
                    inlineScript: |
                      az ml model update \
                        --name "${{ parameters.modelName }}" \
                        --version latest \
                        --set tags.deployment_status=go_prod \
                        --set tags.validation=validated \
                        --workspace-name mlops-workspace \
                        --resource-group rg-mlops-production

                      curl -X PATCH \
                        "${{ variables.apiUrl }}/runs/${{ parameters.runId }}/status?status=completed"
```

### 8.3. Environment avec Approval Gate

La gate manuelle est configurée via un **Azure DevOps Environment** :

```
Azure DevOps → Project Settings → Environments → mlops-production
  └── Approvals and checks
       ├── Approval: Équipe ML (1 approbateur minimum)
       ├── Timeout: 72h
       └── Instructions: "Vérifier le MLScore et les résultats RAGAS
                          dans Azure ML Studio avant d'approuver."
```

Quand le pipeline atteint le stage `ManualValidation`, il **s'arrête et attend** l'approbation dans Azure DevOps. L'approbateur reçoit une notification (email + Teams si connecteur activé).

### 8.4. Déclenchement du pipeline

Le Release Worker (Container Apps Job) déclenche le pipeline via l'API REST Azure DevOps :

```python
import requests

DEVOPS_ORG = os.getenv("AZURE_DEVOPS_ORG")
DEVOPS_PROJECT = os.getenv("AZURE_DEVOPS_PROJECT")
DEVOPS_PAT = os.getenv("AZURE_DEVOPS_PAT")
PIPELINE_ID = os.getenv("AZURE_DEVOPS_PIPELINE_ID")

def trigger_release_pipeline(run_id: int, model_name: str, model_id: str,
                              experiment_name: str, task_type: str):
    url = (f"https://dev.azure.com/{DEVOPS_ORG}/{DEVOPS_PROJECT}"
           f"/_apis/pipelines/{PIPELINE_ID}/runs?api-version=7.1")

    payload = {
        "templateParameters": {
            "runId": str(run_id),
            "modelName": model_name,
            "modelId": model_id,
            "experimentName": experiment_name,
            "taskType": task_type,
        }
    }
    response = requests.post(
        url,
        json=payload,
        auth=("", DEVOPS_PAT),
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()
```

---

## 9. Azure Logic Apps — Notifications Teams

Remplace le `NotifyTeams.py` du plugin Digital.ai Release et le Teams Worker.

### 9.1. Pourquoi Logic Apps

| Critère | Webhook direct | Azure Logic Apps |
|---|---|---|
| Connecteur Teams | HTTP POST manuel | Connecteur natif (Adaptive Cards) |
| Déclencheur | Code custom | Service Bus trigger intégré |
| Retry/error handling | À implémenter | Natif |
| Maintenance | Code Python | Designer visuel (low-code) |
| Coût | 0 (self-hosted) | ~0.001€ par exécution |

### 9.2. Workflow Logic App

```
Trigger: Azure Service Bus → Topic "mlops-events"
                              Subscription "sub-notifications"
    │
    ▼
Parse JSON (body du message)
    │
    ├── Condition: event_type == "run.eval_passed"
    │   └── Action: Post Adaptive Card to Teams
    │       ├── Title: "MLOps — Modèle prêt à tester"
    │       ├── Facts: Run ID, Modèle, MLScore, Expérience
    │       └── Action Button: "Voir dans Azure ML Studio"
    │
    ├── Condition: event_type == "run.eval_rejected"
    │   └── Action: Post Adaptive Card to Teams
    │       ├── Title: "MLOps — Modèle rejeté"
    │       ├── Facts: Run ID, Modèle, MLScore, Seuil
    │       └── Style: Attention (rouge)
    │
    └── Condition: event_type == "run.go_prod"
        └── Action: Post Adaptive Card to Teams
            ├── Title: "MLOps — Modèle en production"
            ├── Facts: Run ID, Modèle, Version
            └── Style: Good (vert)
```

### 9.3. Adaptive Card Teams (modèle)

```json
{
  "type": "AdaptiveCard",
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.4",
  "body": [
    {
      "type": "TextBlock",
      "size": "Large",
      "weight": "Bolder",
      "text": "MLOps — ${modelName}",
      "style": "heading"
    },
    {
      "type": "TextBlock",
      "text": "${message}",
      "wrap": true
    },
    {
      "type": "FactSet",
      "facts": [
        { "title": "Run ID", "value": "${runId}" },
        { "title": "Modèle", "value": "${modelId}" },
        { "title": "MLScore", "value": "${mlScore}" },
        { "title": "Seuil", "value": "${threshold}" },
        { "title": "Expérience", "value": "${experimentName}" }
      ]
    }
  ],
  "actions": [
    {
      "type": "Action.OpenUrl",
      "title": "Ouvrir Azure ML Studio",
      "url": "https://ml.azure.com/experiments/${experimentName}"
    },
    {
      "type": "Action.OpenUrl",
      "title": "Approuver dans Azure DevOps",
      "url": "https://dev.azure.com/${org}/${project}/_environments"
    }
  ]
}
```

---

## 10. Azure Database for PostgreSQL

Remplace le container PostgreSQL 16.

### 10.1. Configuration

| Propriété | Valeur |
|---|---|
| Service | Azure Database for PostgreSQL — Flexible Server |
| Nom | `psql-mlops-production` |
| SKU | Burstable B1ms (1 vCPU, 2 GiB RAM) |
| Stockage | 32 GiB (auto-grow) |
| Version PostgreSQL | 16 |
| Backup | Automatique, rétention 7 jours |
| Networking | VNet Integration (même VNet que Container Apps) |

### 10.2. Bases de données

| Base | Usage |
|---|---|
| `mlops` | Application (datasets, pipeline_runs, run_results) |

> Note : la base `mlflow` n'est plus nécessaire. Azure ML Workspace gère son propre backend de tracking.

### 10.3. Chaîne de connexion

Référencée depuis Key Vault, injectée via Managed Identity :

```
postgresql://{admin}:{password}@psql-mlops-production.postgres.database.azure.com:5432/mlops?sslmode=require
```

---

## 11. Azure Blob Storage — Données et artefacts

Remplace les volumes Docker (`model_outputs`, `mlflow_artifacts`, `./data`).

### 11.1. Structure des containers Blob

| Container Blob | Contenu | Accès |
|---|---|---|
| `datasets` | Fichiers JSONL d'entraînement et d'évaluation | Lecture par tous les workers |
| `model-outputs` | Modèles entraînés (poids, tokenizer, config) | Écriture par training, lecture par eval |
| `mlflow-artifacts` | (géré automatiquement par Azure ML) | Via Azure ML SDK |

### 11.2. Structure des fichiers

```
datasets/
├── train/
│   ├── rag_qa_train.jsonl
│   ├── medical_qa_train.jsonl
│   └── legal_qa_train.jsonl
└── eval/
    ├── ragas_eval.jsonl
    ├── medical_ragas_eval.jsonl
    └── legal_ragas_eval.jsonl

model-outputs/
└── mistral-7b-rag-qa/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

### 11.3. Accès via Azure ML Datastore

```python
from azure.ai.ml.entities import AzureBlobDatastore

datastore = AzureBlobDatastore(
    name="mlops_data",
    account_name="stmlopsproduction",
    container_name="datasets",
    credentials=None,  # Managed Identity
)
ml_client.datastores.create_or_update(datastore)
```

---

## 12. Azure Key Vault — Secrets

Remplace le fichier `.env`.

### 12.1. Secrets stockés

| Secret | Usage |
|---|---|
| `database-url` | Connexion PostgreSQL |
| `servicebus-connection` | Connexion Azure Service Bus |
| `azure-ai-api-key` | Clé API Azure AI Services (Mistral) |
| `azure-devops-pat` | Personal Access Token Azure DevOps |
| `hf-token` | Token HuggingFace Hub (optionnel) |

### 12.2. Accès via Managed Identity

Aucun secret n'est stocké en variable d'environnement en clair. Les Container Apps et Azure ML Jobs accèdent aux secrets via :

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://kv-mlops-production.vault.azure.net", credential=credential)

database_url = client.get_secret("database-url").value
```

Ou via la référence directe dans la configuration Container Apps :

```yaml
secrets:
  - name: database-url
    keyVaultUrl: https://kv-mlops-production.vault.azure.net/secrets/database-url
    identity: system
```

---

## 13. Azure Monitor — Observabilité

Remplace les logs Docker (`docker compose logs`).

### 13.1. Composants

| Composant | Usage |
|---|---|
| **Application Insights** | Traces, requêtes HTTP, exceptions, dépendances (API + UI) |
| **Log Analytics Workspace** | Agrégation de tous les logs (Container Apps, ML Jobs, Service Bus) |
| **Azure Monitor Alerts** | Alertes sur échecs de pipeline, MLScore faible, erreurs API |
| **Dashboards** | KPI en temps réel (runs actifs, MLScore moyen, taux d'échec) |

### 13.2. Alertes recommandées

| Alerte | Condition | Sévérité | Action |
|---|---|---|---|
| Training job échoué | AzureML Job status = Failed | 2 (Warning) | Email + Teams |
| Évaluation timeout | Container Apps Job duration > 3600s | 2 (Warning) | Email |
| API erreurs 5xx | HTTP 5xx count > 5 en 5 min | 1 (Error) | Email + Teams |
| MLScore dégradé | Moyenne MLScore < 0.5 sur 5 derniers runs | 3 (Information) | Email |
| Service Bus dead-letter | Dead-letter count > 0 | 2 (Warning) | Email |

### 13.3. Dashboard Azure

```
┌──────────────────────────────────────────────────────────────┐
│                    MLOps Operations Dashboard                 │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Runs actifs    │  │  MLScore moyen  │  │ Taux succès  │ │
│  │      3          │  │     0.742       │  │    87%       │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  MLScore par run (30 derniers jours)                    │ │
│  │  ████████████ 0.81                                      │ │
│  │  █████████ 0.67                                         │ │
│  │  ███████████ 0.74                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Durée moyenne par étape                                │ │
│  │  Training:    45 min   Evaluation:  12 min              │ │
│  │  Approval:    4.2 h    Total:       5.1 h               │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 14. Sécurité et identité

### 14.1. Managed Identity

Toutes les communications inter-services utilisent **Azure Managed Identity** (System-Assigned) :

| Service | Rôle RBAC | Ressource cible |
|---|---|---|
| Container App (API) | Key Vault Secrets User | Key Vault |
| Container App (API) | Azure Service Bus Data Sender | Service Bus |
| Container App (API) | Storage Blob Data Reader | Blob Storage |
| Container Apps Job (Eval) | Key Vault Secrets User | Key Vault |
| Container Apps Job (Eval) | Azure Service Bus Data Receiver | Service Bus |
| Container Apps Job (Eval) | Storage Blob Data Contributor | Blob Storage |
| Azure ML Compute | AzureML Data Scientist | ML Workspace |
| Azure ML Compute | Storage Blob Data Contributor | Blob Storage |
| Logic App | Azure Service Bus Data Receiver | Service Bus |

### 14.2. Réseau

```
┌──────────────────────────────────────────────┐
│          VNet: vnet-mlops-production          │
│          Address space: 10.0.0.0/16          │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Subnet: snet-container-apps             │ │
│  │ 10.0.1.0/24                             │ │
│  │ → Container Apps Environment            │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Subnet: snet-postgres                   │ │
│  │ 10.0.2.0/24                             │ │
│  │ → PostgreSQL Flexible Server            │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Subnet: snet-ml-compute                 │ │
│  │ 10.0.3.0/24                             │ │
│  │ → Azure ML Compute Cluster              │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  Private Endpoints:                          │
│  → Key Vault                                 │
│  → Storage Account                           │
│  → Service Bus                               │
│  → Azure ML Workspace                        │
└──────────────────────────────────────────────┘
```

### 14.3. Pas de secrets dans le code

| Actuel (.env) | Azure (Key Vault + Managed Identity) |
|---|---|
| `MISTRAL_API_KEY=xxx` | Secret `azure-ai-api-key` dans Key Vault |
| `DATABASE_URL=postgresql://mlops:mlops@...` | Secret `database-url` dans Key Vault |
| `HF_TOKEN=hf_xxx` | Secret `hf-token` dans Key Vault |
| Variables en clair dans docker-compose | Références Key Vault dans Container Apps config |

---

## 15. Infrastructure as Code (Bicep)

### 15.1. Structure des fichiers

```
infra/
├── main.bicep                    # Point d'entrée, orchestration des modules
├── parameters.prod.json          # Paramètres de production
├── parameters.dev.json           # Paramètres de développement
└── modules/
    ├── resource-group.bicep
    ├── vnet.bicep                # VNet + subnets + NSGs
    ├── postgres.bicep            # Azure DB for PostgreSQL
    ├── storage.bicep             # Storage Account + containers
    ├── keyvault.bicep            # Key Vault + secrets
    ├── servicebus.bicep          # Service Bus + topics + subscriptions
    ├── container-registry.bicep  # ACR
    ├── container-apps-env.bicep  # Container Apps Environment
    ├── container-app-api.bicep   # Container App API
    ├── container-app-ui.bicep    # Container App UI
    ├── container-app-jobs.bicep  # Container Apps Jobs (eval, release)
    ├── ml-workspace.bicep        # Azure ML Workspace + Compute
    ├── logic-app.bicep           # Logic App (notifications)
    └── monitor.bicep             # Application Insights + Alerts
```

### 15.2. Extrait `main.bicep`

```bicep
targetScope = 'subscription'

param environment string = 'prod'
param location string = 'westeurope'

resource rg 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-mlops-${environment}'
  location: location
}

module vnet 'modules/vnet.bicep' = {
  name: 'vnet'
  scope: rg
  params: { location: location, environment: environment }
}

module postgres 'modules/postgres.bicep' = {
  name: 'postgres'
  scope: rg
  params: {
    location: location
    subnetId: vnet.outputs.postgresSubnetId
    administratorLogin: 'mlopsadmin'
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  scope: rg
  params: { location: location }
}

module keyvault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  scope: rg
  params: { location: location }
}

module servicebus 'modules/servicebus.bicep' = {
  name: 'servicebus'
  scope: rg
  params: { location: location }
}

module mlWorkspace 'modules/ml-workspace.bicep' = {
  name: 'ml-workspace'
  scope: rg
  params: {
    location: location
    storageAccountId: storage.outputs.storageAccountId
    keyVaultId: keyvault.outputs.keyVaultId
  }
}

module containerApps 'modules/container-apps-env.bicep' = {
  name: 'container-apps'
  scope: rg
  params: {
    location: location
    subnetId: vnet.outputs.containerAppsSubnetId
  }
}
```

### 15.3. Déploiement

```bash
# Déployer l'infrastructure
az deployment sub create \
  --location westeurope \
  --template-file infra/main.bicep \
  --parameters infra/parameters.prod.json

# Construire et pousser les images
az acr build --registry acr-mlops --image mlops-api:latest --file docker/Dockerfile.api .
az acr build --registry acr-mlops --image mlops-ui:latest --file docker/Dockerfile.ui .
az acr build --registry acr-mlops --image mlops-eval:latest --file docker/Dockerfile.evaluation .
```

---

## 16. Workflow de bout en bout

### Schéma complet du flux Azure

```
┌──────────┐  HTTPS   ┌──────────────┐  publish   ┌──────────────────┐
│Streamlit │────────►│   FastAPI    │──────────►│ Azure Service Bus │
│Container │         │  Container   │            │  Topic: mlops-    │
│  App     │         │    App       │            │  events           │
└──────────┘         └──────┬───────┘            └────────┬──────────┘
                            │                             │
                      ┌─────┴─────┐          ┌────────────┼────────────────┐
                      │           │          │            │                │
                      ▼           │          ▼            ▼                ▼
              ┌──────────────┐    │   ┌────────────┐ ┌─────────┐  ┌────────────┐
              │ Azure DB for │    │   │ Container  │ │ Release │  │ Azure      │
              │ PostgreSQL   │    │   │ Apps Job   │ │ Worker  │  │ Logic App  │
              │ (runs, data) │    │   │ (Eval)     │ │ (Job)   │  │ (→ Teams)  │
              └──────────────┘    │   └─────┬──────┘ └────┬────┘  └────────────┘
                                  │         │             │
                                  │         ▼             ▼
                                  │   ┌──────────┐  ┌──────────────────┐
                                  │   │Azure AI  │  │ Azure DevOps     │
                                  │   │Services  │  │ Pipeline         │
                                  │   │(Mistral) │  │ (4 stages +      │
                                  │   └──────────┘  │  approval gate)  │
                                  │                  └────────┬─────────┘
                                  │                           │
                                  ▼                           ▼
                           ┌─────────────────────────────────────────┐
                           │        Azure Machine Learning           │
                           │                                         │
                           │  ┌─────────────┐  ┌──────────────────┐ │
                           │  │ Experiments │  │  Model Registry  │ │
                           │  │ (tracking)  │  │  (tags, stages)  │ │
                           │  └─────────────┘  └──────────────────┘ │
                           │                                         │
                           │  ┌─────────────────────────────────┐   │
                           │  │  ML Compute Cluster (Training)  │   │
                           │  │  GPU auto-scale 0 → N           │   │
                           │  └─────────────────────────────────┘   │
                           └─────────────────────────────────────────┘
```

### Séquence détaillée

| # | Action | Service Azure | Détail |
|---|---|---|---|
| 1 | L'utilisateur lance un pipeline | Container App (Streamlit) | POST vers l'API |
| 2 | L'API crée le run en BDD | Container App (FastAPI) + PostgreSQL | Status = `pending` |
| 3 | L'API publie `run.created` | Azure Service Bus | Topic `mlops-events` |
| 4 | Le Release Worker consomme le message | Container Apps Job | Déclenche le pipeline Azure DevOps |
| 5 | Stage 1 : chargement du modèle | Azure DevOps Pipeline | `huggingface_hub` + `az ml model create` |
| 6 | Stage 2 : soumission du training | Azure DevOps → Azure ML | `az ml job create` sur GPU cluster |
| 7 | Le job de training s'exécute | Azure ML Compute | HuggingFace + LoRA + MLflow tracking |
| 8 | Training terminé, status → `evaluating` | API + Service Bus | Publie `run.training_done` |
| 9 | Stage 3 : évaluation RAGAS | Container Apps Job (Eval) | Azure AI (Mistral) comme LLM juge |
| 10 | Calcul du MLScore | Container Apps Job | Moyenne pondérée des métriques |
| 11a | MLScore ≥ seuil | Azure DevOps Pipeline | → Stage 4b : Validation manuelle |
| 11b | MLScore < seuil | Azure DevOps Pipeline | → Stage 4a : Tag `rejected`, notification Teams |
| 12 | Notification Teams | Azure Logic App | Adaptive Card avec lien Azure ML Studio |
| 13 | Gate manuelle : attente d'approbation | Azure DevOps Environment | Approval check configuré |
| 14 | Approbateur valide | Azure DevOps | → Tag `go_prod` sur le modèle dans Azure ML Registry |

---

## 17. Estimation des coûts

### Coût mensuel estimé (usage modéré : ~10 runs/mois)

| Service | SKU | Estimation mensuelle |
|---|---|---|
| Azure DB for PostgreSQL | Burstable B1ms | ~15 € |
| Azure Container Apps (API + UI) | Consumption (2 apps) | ~5 € |
| Azure Container Apps Jobs (Eval + Release) | Consumption | ~3 € |
| Azure Service Bus | Standard | ~10 € |
| Azure ML Workspace | Standard | ~0 € (gratuit, coût = compute) |
| Azure ML Compute (GPU) | Standard_NC6s_v3 (scale-to-zero) | ~50 € (10 runs × ~30 min GPU) |
| Azure Blob Storage | Standard LRS | ~2 € |
| Azure Key Vault | Standard | ~1 € |
| Azure Logic Apps | Consumption | ~0.10 € |
| Azure Container Registry | Basic | ~5 € |
| Azure AI Services (Mistral) | Pay-as-you-go | ~20 € (évaluations RAGAS) |
| Application Insights | Log Analytics | ~5 € |
| Azure DevOps | Basic (1 parallel job) | Gratuit |
| **TOTAL** | | **~116 €/mois** |

### Comparaison avec l'architecture Docker

| Aspect | Docker Compose (self-hosted) | Azure Native |
|---|---|---|
| Coût infrastructure | Machine(s) à provisionner (~200–500€/mois pour GPU) | ~116 €/mois (pay-as-you-go) |
| GPU idle | Payé 24/7 si réservé | Scale-to-zero (payé uniquement pendant le training) |
| Maintenance | OS, Docker, sécurité, backups manuels | Zéro (PaaS managé) |
| Haute disponibilité | À configurer manuellement | Incluse (SLA 99.9%+) |
| Scalabilité | Limité à la machine | Auto-scale sur demande |
| Sécurité réseau | Docker network (pas chiffré) | VNet + Private Endpoints + TLS |
| Secrets | Fichier `.env` | Key Vault + Managed Identity |
| Monitoring | `docker compose logs` | Azure Monitor, alertes, dashboards |

---

## 18. Plan de migration

### Phase 1 — Fondations (Semaine 1-2)

1. Créer le Resource Group et déployer le VNet (Bicep)
2. Déployer Azure DB for PostgreSQL + migrer le schéma
3. Déployer Azure Storage + uploader les datasets
4. Déployer Azure Key Vault + créer les secrets
5. Déployer Azure Container Registry + build les images

### Phase 2 — API et UI (Semaine 2-3)

6. Déployer Container Apps Environment
7. Déployer Container App API (FastAPI)
8. Adapter l'API pour Azure Service Bus (publication d'événements)
9. Déployer Container App UI (Streamlit)
10. Tester la chaîne UI → API → PostgreSQL

### Phase 3 — ML et Training (Semaine 3-4)

11. Créer Azure ML Workspace
12. Configurer le MLflow Tracking URI vers Azure ML
13. Créer le Compute Cluster (GPU)
14. Adapter le training worker pour Azure ML Jobs
15. Déployer le dispatcher (Container Apps Job)
16. Tester un training de bout en bout

### Phase 4 — Évaluation et AI (Semaine 4-5)

17. Déployer Mistral sur Azure AI Services
18. Adapter l'evaluation worker pour Azure AI endpoints
19. Déployer le Container Apps Job (Eval) avec trigger Service Bus
20. Tester l'évaluation RAGAS sur Azure

### Phase 5 — Orchestration et Release (Semaine 5-6)

21. Créer le pipeline Azure DevOps (YAML)
22. Configurer l'Environment avec Approval Gate
23. Déployer le Release Worker (Container Apps Job)
24. Créer la Logic App pour les notifications Teams
25. Tester le workflow complet : UI → Training → Eval → Approval → Tag

### Phase 6 — Monitoring et Hardening (Semaine 6-7)

26. Configurer Application Insights sur API et UI
27. Créer les alertes Azure Monitor
28. Créer le dashboard opérationnel
29. Activer les Private Endpoints
30. Tests de charge et validation sécurité
