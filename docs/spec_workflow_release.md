# Spécification — Workflow MLOps avec orchestration de release

> Spécification détaillée du workflow MLOps intégrant RabbitMQ, Digital.ai Release, Microsoft Teams et MLflow Model Registry pour l'automatisation du cycle de vie des modèles LLM.

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Architecture actuelle](#2-architecture-actuelle)
3. [Architecture cible](#3-architecture-cible)
4. [Workflow détaillé](#4-workflow-détaillé)
5. [Composants nouveaux](#5-composants-nouveaux)
6. [Plugin Digital.ai Release](#6-plugin-digitalai-release)
7. [Intégration RabbitMQ](#7-intégration-rabbitmq)
8. [Intégration Microsoft Teams](#8-intégration-microsoft-teams)
9. [Gestion des tags MLflow](#9-gestion-des-tags-mlflow)
10. [Variables d'environnement](#10-variables-denvironnement)
11. [Infrastructure Docker](#11-infrastructure-docker)
12. [Matrice des décisions](#12-matrice-des-décisions)
13. [Comparaison Jira vs Digital.ai Release](#13-comparaison-jira-vs-digitalai-release)

---

## 1. Contexte et objectif

### Situation actuelle

Le système MLOps actuel fonctionne en mode **poll-based** :

- L'UI Streamlit envoie une requête `POST /runs` à l'API FastAPI.
- Les workers (training, evaluation) interrogent l'API à intervalle régulier pour détecter les runs à traiter.
- Les changements de statut sont enregistrés dans PostgreSQL et consultables via l'UI.
- Aucune notification externe ni orchestration de release n'existe.

### Objectif

Mettre en place un **workflow de release automatisé** qui :

1. Déclenche la création d'une release traçable à chaque lancement de pipeline.
2. Fait progresser automatiquement la release à chaque étape (chargement, entraînement, évaluation).
3. Applique une **gate automatique** basée sur le MLScore pour valider ou rejeter le modèle.
4. Propose une **gate manuelle** permettant à un humain de valider le passage en production.
5. Notifie l'équipe via **Microsoft Teams** aux étapes clés.
6. Applique automatiquement les tags `go_prod` ou `rejected` dans **MLflow Model Registry**.

---

## 2. Architecture actuelle

```
┌──────────┐    POST /runs    ┌─────────┐    PostgreSQL    ┌──────────┐
│    UI    │─────────────────►│   API   │◄────────────────►│    DB    │
│Streamlit │                  │ FastAPI │                   │PostgreSQL│
└──────────┘                  └────┬────┘                   └──────────┘
                                   │
                          poll (HTTP GET)
                        ┌──────────┴──────────┐
                        ▼                     ▼
                  ┌───────────┐         ┌───────────┐       ┌──────────┐
                  │ Training  │         │Evaluation │──────►│  MLflow  │
                  │  Worker   │────────►│  Worker   │       │ Tracking │
                  └───────────┘         └───────────┘       └──────────┘
                       │
                 status: pending
                    → training
                    → evaluating
                    → completed | failed
```

### Limites identifiées

| Limite | Impact |
|---|---|
| Polling HTTP uniquement | Latence de détection (= `POLL_INTERVAL`, 10s par défaut) |
| Aucune notification externe | L'équipe doit consulter l'UI manuellement |
| Pas de gate manuelle | Aucune validation humaine avant mise en production |
| Pas de traçabilité de release | Pas d'historique formel des décisions de déploiement |
| Tags MLflow manuels | Le passage en production n'est pas automatisé |

---

## 3. Architecture cible

```
┌──────────┐   POST /runs   ┌─────────┐  publish event  ┌──────────┐
│    UI    │───────────────►│   API   │────────────────►│ RabbitMQ │
│Streamlit │                │ FastAPI │                  └─────┬────┘
└──────────┘                └─────────┘                        │
                                                               │
                       ┌───────────────────────────────────────┤
                       │                                       │
                       ▼                                       ▼
              ┌─────────────────┐                    ┌───────────────┐
              │  Digital.ai     │                    │   Training    │
              │  Release Worker │                    │    Worker     │
              │  (consomme      │                    │  (consomme    │
              │   run.created)  │                    │   run.created)│
              └────────┬────────┘                    └───────┬───────┘
                       │                                     │
                       ▼                                     ▼
         ╔══════════════════════════╗               ┌───────────────┐
         ║  DIGITAL.AI RELEASE     ║               │  Evaluation   │
         ║  Pipeline orchestré     ║◄──────────────│    Worker     │
         ║  (phases, gates, tasks) ║  events via   └───────────────┘
         ╚════════════╤═══════════╝  RabbitMQ
                      │
              ┌───────┴───────┐
              ▼               ▼
        ┌──────────┐   ┌──────────┐
        │  MLflow  │   │Microsoft │
        │ Registry │   │  Teams   │
        │(tag/stage)│  │(webhook) │
        └──────────┘   └──────────┘
```

---

## 4. Workflow détaillé

### 4.1. Schéma des états

```
                         ┌──────────────────────────────────────────────────────────────┐
                         │              DIGITAL.AI RELEASE — PHASES                     │
                         │                                                              │
                         │  Chargement │ Entraînement │ Évaluation │ Validation │ Clos  │
                         │   modèle   │              │            │  manuelle  │       │
                         └──────┬──────┴──────┬───────┴─────┬──────┴─────┬──────┴───┬───┘
                                │             │             │            │          │
  UI Streamlit                  │             │             │            │          │
  └─► POST /runs ──► RabbitMQ  │             │             │            │          │
                         │      │             │             │            │          │
                         ▼      │             │             │            │          │
                    Création    │             │             │            │          │
                    release ────┘             │             │            │          │
                         │                    │             │            │          │
                    Load model                │             │            │          │
                    Register MLflow           │             │            │          │
                         │                    │             │            │          │
                    ─────┼────────────────────┘             │            │          │
                         │                                  │            │          │
                    Trigger training                        │            │          │
                    Wait for completion                     │            │          │
                         │                                  │            │          │
                    ─────┼──────────────────────────────────┘            │          │
                         │                                               │          │
                    Trigger evaluation                                   │          │
                    Calcul MLScore                                       │          │
                         │                                               │          │
                    ┌────┴────┐                                          │          │
                    │         │                                          │          │
               ≥ seuil    < seuil                                       │          │
                    │         │                                          │          │
                    │         └── tag "rejected" ───────────────────────►│──► Clos  │
                    │             Notify Teams (rejet)                   │          │
                    │                                                    │          │
                    └── Notify Teams ────────────────────────────────────┘          │
                        (modèle prêt à tester)                                     │
                              │                                                    │
                         GATE MANUELLE                                             │
                         (approbation humaine)                                     │
                              │                                                    │
                         ┌────┴────┐                                               │
                         │         │                                               │
                    Approuvé    Rejeté                                              │
                         │         │                                               │
                    tag "go_prod"  tag "rejected"                                  │
                    stage →        stage →                                          │
                    Production     Archived ────────────────────────────────────────┘
```

### 4.2. Tableau des transitions

| # | Déclencheur | Action système | Phase Release | MLflow | Teams |
|---|---|---|---|---|---|
| 1 | L'utilisateur clique "Lancer" dans Streamlit | `POST /runs` → RabbitMQ publie `run.created` | Release créée → **Chargement modèle** | — | — |
| 2 | Release Worker consomme `run.created` | Télécharge le modèle HuggingFace, enregistre dans le registry MLflow | Phase 1 en cours | Modèle enregistré (version initiale) | — |
| 3 | Chargement terminé | Release avance automatiquement | → **Entraînement** | — | — |
| 4 | Training Worker consomme l'événement | Fine-tuning LoRA, logging des métriques d'entraînement | Phase 2 en cours | `train_loss`, `perplexity` loggés | — |
| 5 | Training terminé | Worker publie `run.training_done` | → **Évaluation** | Métriques finales loggées | — |
| 6 | Evaluation Worker consomme `run.training_done` | Inférence + évaluation RAGAS + calcul MLScore | Phase 3 en cours | Métriques RAGAS loggées | — |
| 7a | MLScore ≥ seuil | Tâche `CheckMLScore` valide | → **Validation manuelle** | — | Adaptive Card : "Modèle prêt à tester" |
| 7b | MLScore < seuil | Tâche `CheckMLScore` rejette | → **Clos** (abort) | Tag `rejected`, stage `Archived` | Adaptive Card : "Modèle rejeté" |
| 8 | Approbateur valide la gate | Tâche `TagModel` exécutée | → **Clos** (succès) | Tag `go_prod`, stage `Production` | Adaptive Card : "Modèle en production" |
| 9 | Approbateur rejette la gate | Tâche `TagModel` exécutée | → **Clos** (rejet manuel) | Tag `rejected`, stage `Archived` | Adaptive Card : "Modèle rejeté manuellement" |

### 4.3. Calcul du MLScore

Moyenne pondérée des métriques RAGAS :

| Métrique | Poids |
|---|---|
| `faithfulness` | 0.30 |
| `answer_relevancy` | 0.20 |
| `context_precision` | 0.25 |
| `context_recall` | 0.25 |

Formule :

```
MLScore = Σ (métrique_i × poids_i)
```

Seuil de validation par défaut : **0.7** (configurable via `MLSCORE_THRESHOLD`).

---

## 5. Composants nouveaux

### 5.1. RabbitMQ (broker de messages)

Remplace le mécanisme de polling HTTP par un modèle événementiel (pub/sub).

| Propriété | Valeur |
|---|---|
| Image Docker | `rabbitmq:3-management` |
| Port AMQP | 5672 |
| Port UI Management | 15672 |
| Exchange | `mlops.events` (type `topic`) |
| Credentials par défaut | `guest:guest` (à changer en production) |

### 5.2. Digital.ai Release Worker (nouveau container)

Service Python qui :

1. Consomme les événements `run.created` depuis RabbitMQ.
2. Crée une Release dans Digital.ai Release à partir d'un template prédéfini.
3. Transmet le `run_id` et les métadonnées en tant que variables de release.

### 5.3. Plugin Python Digital.ai Release (nouveau)

Ensemble de tâches custom déployées dans Digital.ai Release Server.

---

## 6. Plugin Digital.ai Release

### 6.1. Structure du plugin

```
digitalai-mlops-plugin/
├── synthetic.xml                 # Déclaration des types de tâches
├── plugin-version.properties     # Métadonnées (version, nom)
└── mlops/
    ├── __init__.py
    ├── LoadModel.py              # Télécharge le modèle HuggingFace
    ├── RegisterToMLflow.py       # Publie dans MLflow Model Registry
    ├── TriggerTraining.py        # Déclenche l'entraînement via API
    ├── WaitForTraining.py        # Polling jusqu'à fin du training
    ├── TriggerEvaluation.py      # Déclenche l'évaluation via API
    ├── CheckMLScore.py           # Vérifie le seuil, branche le pipeline
    ├── TagModel.py               # Applique un tag dans MLflow Registry
    └── NotifyTeams.py            # Envoie une Adaptive Card Teams
```

### 6.2. Déclaration des tâches (`synthetic.xml`)

```xml
<synthetic xmlns="http://www.xebialabs.com/deployit/synthetic"
           xmlns:type="http://www.xebialabs.com/deployit/synthetic">

  <!-- Connexion partagée vers la stack MLOps -->
  <type type="mlops.Server" extends="configuration.HttpConnection">
    <property name="mlflowUrl"        category="MLflow"       required="true" />
    <property name="mlscoreThreshold" category="Validation"   default="0.7" kind="string" />
    <property name="teamsWebhookUrl"  category="Notifications" required="false" />
  </type>

  <!-- Tâche : Chargement du modèle -->
  <type type="mlops.LoadModel" extends="xlrelease.PythonScript">
    <property name="server"    category="input"  referenced-type="mlops.Server" kind="ci" />
    <property name="modelId"   category="input"  required="true"
              description="HuggingFace model ID (ex: mistralai/Mistral-7B-v0.1)" />
    <property name="outputDir" category="input"  default="/app/outputs" />
    <property name="cachePath" category="output" />
  </type>

  <!-- Tâche : Enregistrement MLflow -->
  <type type="mlops.RegisterToMLflow" extends="xlrelease.PythonScript">
    <property name="server"          category="input"  referenced-type="mlops.Server" kind="ci" />
    <property name="runId"           category="input"  required="true" kind="integer" />
    <property name="modelPath"       category="input"  required="true" />
    <property name="mlflowRunId"     category="output" />
    <property name="mlflowModelName" category="output" />
    <property name="mlflowVersion"   category="output" />
  </type>

  <!-- Tâche : Déclenchement entraînement -->
  <type type="mlops.TriggerTraining" extends="xlrelease.PythonScript">
    <property name="server" category="input" referenced-type="mlops.Server" kind="ci" />
    <property name="runId"  category="input" required="true" kind="integer" />
  </type>

  <!-- Tâche : Attente fin entraînement (polling) -->
  <type type="mlops.WaitForTraining" extends="xlrelease.PythonScript">
    <property name="server"       category="input"  referenced-type="mlops.Server" kind="ci" />
    <property name="runId"        category="input"  required="true" kind="integer" />
    <property name="pollInterval" category="input"  default="30" kind="integer"
              description="Intervalle de polling en secondes" />
    <property name="trainLoss"    category="output" kind="string" />
    <property name="perplexity"   category="output" kind="string" />
  </type>

  <!-- Tâche : Déclenchement évaluation -->
  <type type="mlops.TriggerEvaluation" extends="xlrelease.PythonScript">
    <property name="server" category="input" referenced-type="mlops.Server" kind="ci" />
    <property name="runId"  category="input" required="true" kind="integer" />
  </type>

  <!-- Tâche : Vérification MLScore + décision -->
  <type type="mlops.CheckMLScore" extends="xlrelease.PythonScript">
    <property name="server"    category="input"  referenced-type="mlops.Server" kind="ci" />
    <property name="runId"     category="input"  required="true" kind="integer" />
    <property name="threshold" category="input"  default="0.7" kind="string" />
    <property name="mlScore"   category="output" kind="string" />
    <property name="passed"    category="output" kind="boolean" />
  </type>

  <!-- Tâche : Tagging modèle MLflow -->
  <type type="mlops.TagModel" extends="xlrelease.PythonScript">
    <property name="server" category="input" referenced-type="mlops.Server" kind="ci" />
    <property name="runId"  category="input" required="true" kind="integer" />
    <property name="tag"    category="input" required="true"
              description="Tag à appliquer : go_prod | rejected" />
  </type>

  <!-- Tâche : Notification Teams -->
  <type type="mlops.NotifyTeams" extends="xlrelease.PythonScript">
    <property name="server"  category="input" referenced-type="mlops.Server" kind="ci" />
    <property name="runId"   category="input" required="true" kind="integer" />
    <property name="message" category="input" required="true" />
  </type>

</synthetic>
```

### 6.3. Template de release

Le template Digital.ai Release se structure en 4 phases :

#### Phase 1 — Chargement & Publication

| Tâche | Type | Entrées | Sorties |
|---|---|---|---|
| Télécharger le modèle | `mlops.LoadModel` | `modelId`, `outputDir` | `cachePath` |
| Publier dans MLflow | `mlops.RegisterToMLflow` | `runId`, `modelPath` | `mlflowRunId`, `mlflowModelName`, `mlflowVersion` |

#### Phase 2 — Entraînement

| Tâche | Type | Entrées | Sorties |
|---|---|---|---|
| Lancer l'entraînement | `mlops.TriggerTraining` | `runId` | — |
| Attendre la fin | `mlops.WaitForTraining` | `runId`, `pollInterval=30` | `trainLoss`, `perplexity` |

#### Phase 3 — Évaluation

| Tâche | Type | Entrées | Sorties |
|---|---|---|---|
| Lancer l'évaluation | `mlops.TriggerEvaluation` | `runId` | — |
| Vérifier le MLScore | `mlops.CheckMLScore` | `runId`, `threshold` | `mlScore`, `passed` |
| (si rejeté) Tagger "rejected" | `mlops.TagModel` | `runId`, `tag=rejected` | — |
| (si rejeté) Notifier Teams | `mlops.NotifyTeams` | `runId`, `message="Modèle rejeté (MLScore insuffisant)"` | — |

#### Phase 4 — Validation manuelle

| Tâche | Type | Entrées | Sorties |
|---|---|---|---|
| Notifier Teams | `mlops.NotifyTeams` | `runId`, `message="Modèle prêt à tester"` | — |
| **Gate d'approbation** | Gate native Digital.ai | Assignée à l'équipe ML | Approuvé / Rejeté |
| (si approuvé) Tagger "go_prod" | `mlops.TagModel` | `runId`, `tag=go_prod` | — |
| (si rejeté) Tagger "rejected" | `mlops.TagModel` | `runId`, `tag=rejected` | — |

### 6.4. Implémentation des tâches Python

#### `mlops/CheckMLScore.py`

```python
import json
import urllib2

api_url = server["url"]
run_id = pythonScript.getProperty("runId")
threshold = float(pythonScript.getProperty("threshold"))

WEIGHTS = {
    "faithfulness": 0.30,
    "answer_relevancy": 0.20,
    "context_precision": 0.25,
    "context_recall": 0.25,
}

request = urllib2.Request("%s/runs/%s" % (api_url, run_id))
response = urllib2.urlopen(request)
run = json.loads(response.read())

results = {r["metric_name"]: r["metric_value"] for r in run["results"]}

ml_score = sum(
    results.get(metric, 0.0) * weight
    for metric, weight in WEIGHTS.items()
)

pythonScript.setProperty("mlScore", str(round(ml_score, 4)))
pythonScript.setProperty("passed", ml_score >= threshold)

if ml_score < threshold:
    raise Exception(
        "MLScore %.4f inferieur au seuil %.2f — modele rejete" % (ml_score, threshold)
    )
```

#### `mlops/WaitForTraining.py`

```python
import json
import time
import urllib2

api_url = server["url"]
run_id = pythonScript.getProperty("runId")
poll_interval = int(pythonScript.getProperty("pollInterval"))

TERMINAL_STATUSES = ("evaluating", "completed", "failed")

while True:
    request = urllib2.Request("%s/runs/%s" % (api_url, run_id))
    response = urllib2.urlopen(request)
    run = json.loads(response.read())

    status = run["status"]
    if status in TERMINAL_STATUSES:
        break
    if status == "failed":
        raise Exception("Training echoue: %s" % run.get("error_message", "inconnu"))

    time.sleep(poll_interval)

results = {r["metric_name"]: r["metric_value"] for r in run["results"]}
pythonScript.setProperty("trainLoss", str(results.get("train_loss", "N/A")))
pythonScript.setProperty("perplexity", str(results.get("perplexity", "N/A")))
```

#### `mlops/TagModel.py`

```python
import json
import urllib2

api_url = server["url"]
mlflow_url = server["mlflowUrl"]
run_id = pythonScript.getProperty("runId")
tag = pythonScript.getProperty("tag")

request = urllib2.Request("%s/runs/%s" % (api_url, run_id))
response = urllib2.urlopen(request)
run = json.loads(response.read())

model_name = run.get("mlflow_model_name")
model_version = run.get("mlflow_model_version")

if model_name and model_version:
    stage = "Production" if tag == "go_prod" else "Archived"
    payload = json.dumps({
        "name": model_name,
        "version": model_version,
        "stage": stage,
    })
    req = urllib2.Request(
        "%s/api/2.0/mlflow/model-versions/transition-stage" % mlflow_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    urllib2.urlopen(req)

    tag_payload = json.dumps({
        "name": model_name,
        "version": model_version,
        "key": "deployment_status",
        "value": tag,
    })
    tag_req = urllib2.Request(
        "%s/api/2.0/mlflow/model-versions/set-tag" % mlflow_url,
        data=tag_payload,
        headers={"Content-Type": "application/json"},
    )
    urllib2.urlopen(tag_req)
```

#### `mlops/NotifyTeams.py`

```python
import json
import urllib2

webhook_url = server["teamsWebhookUrl"]
run_id = pythonScript.getProperty("runId")
message = pythonScript.getProperty("message")
api_url = server["url"]

request = urllib2.Request("%s/runs/%s" % (api_url, run_id))
response = urllib2.urlopen(request)
run = json.loads(response.read())

card = {
    "type": "message",
    "attachments": [{
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "size": "Large",
                    "weight": "Bolder",
                    "text": "MLOps — %s" % run["model_name"],
                },
                {"type": "TextBlock", "text": message, "wrap": True},
                {
                    "type": "FactSet",
                    "facts": [
                        {"title": "Run ID", "value": str(run_id)},
                        {"title": "Modele", "value": run["model_id"]},
                        {"title": "Statut", "value": run["status"]},
                        {"title": "Experience", "value": run["experiment_name"]},
                    ],
                },
            ],
            "actions": [
                {
                    "type": "Action.OpenUrl",
                    "title": "Voir dans MLflow",
                    "url": "%s/#/experiments" % server["mlflowUrl"],
                }
            ],
        },
    }],
}

req = urllib2.Request(
    webhook_url,
    data=json.dumps(card),
    headers={"Content-Type": "application/json"},
)
urllib2.urlopen(req)
```

---

## 7. Intégration RabbitMQ

### 7.1. Topologie des exchanges et queues

| Exchange | Type | Routing Key | Queue(s) consommatrice(s) | Consommateur |
|---|---|---|---|---|
| `mlops.events` | `topic` | `run.created` | `q.release`, `q.training` | Release Worker, Training Worker |
| `mlops.events` | `topic` | `run.training_done` | `q.evaluation`, `q.release.training_done` | Evaluation Worker, Release Worker |
| `mlops.events` | `topic` | `run.eval_passed` | `q.release.eval_passed` | Release Worker |
| `mlops.events` | `topic` | `run.eval_rejected` | `q.release.eval_rejected` | Release Worker |

### 7.2. Format des messages

Tous les messages utilisent le format JSON suivant :

```json
{
  "event": "run.created",
  "run_id": 42,
  "timestamp": "2026-02-22T14:30:00Z",
  "payload": {
    "model_name": "mistral-7b-rag-qa",
    "model_id": "mistralai/Mistral-7B-v0.1",
    "task_type": "finetune",
    "experiment_name": "rag-qa-finetune"
  }
}
```

### 7.3. Migration du polling vers RabbitMQ

Les workers existants passent du mode polling au mode consommateur :

**Avant (polling HTTP) :**

```python
while True:
    runs = httpx.get(f"{API_URL}/runs?status=pending").json()
    for run in runs:
        process_run(run)
    time.sleep(POLL_INTERVAL)
```

**Après (consommation RabbitMQ) :**

```python
import pika
import json

connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
channel = connection.channel()
channel.exchange_declare(exchange="mlops.events", exchange_type="topic", durable=True)
channel.queue_declare(queue="q.training", durable=True)
channel.queue_bind(exchange="mlops.events", queue="q.training", routing_key="run.created")

def on_message(ch, method, properties, body):
    event = json.loads(body)
    run_id = event["run_id"]
    run = httpx.get(f"{API_URL}/runs/{run_id}").json()
    if run["task_type"] == "finetune":
        process_run(run)
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue="q.training", on_message_callback=on_message)
channel.start_consuming()
```

### 7.4. Publication depuis l'API

L'API publie un événement dans RabbitMQ après chaque changement de statut :

```python
# Dans api/main.py — endpoint PATCH /runs/{run_id}/status
def update_run_status(run_id: int, status: str, db: Session = Depends(get_db)):
    run = db.query(PipelineRun).get(run_id)
    run.status = RunStatus(status)
    if status in ("completed", "failed"):
        run.finished_at = datetime.datetime.utcnow()
    db.commit()

    publish_event(f"run.{status}", run_id=run.id, payload={
        "model_name": run.model_name,
        "model_id": run.model_id,
        "task_type": run.task_type,
        "experiment_name": run.experiment_name,
    })

    return {"ok": True}
```

---

## 8. Intégration Microsoft Teams

### 8.1. Mécanisme

Utilisation du **webhook entrant Teams** (Incoming Webhook Connector).

L'URL du webhook est configurée via la variable d'environnement `TEAMS_WEBHOOK_URL` et référencée dans la configuration du serveur `mlops.Server` dans Digital.ai Release.

### 8.2. Événements notifiés

| Événement | Contenu de la notification |
|---|---|
| Modèle prêt à tester (MLScore ≥ seuil) | Nom du modèle, MLScore, lien MLflow, demande d'approbation |
| Modèle rejeté automatiquement (MLScore < seuil) | Nom du modèle, MLScore, détails des métriques |
| Modèle approuvé pour production | Nom du modèle, tag `go_prod`, version MLflow |
| Modèle rejeté manuellement | Nom du modèle, tag `rejected`, identité de l'approbateur |

### 8.3. Format Adaptive Card

Les notifications utilisent le format **Adaptive Card v1.4** pour une présentation riche dans Teams, incluant :

- Titre avec le nom du modèle
- Message contextuel
- Tableau de faits (Run ID, modèle, statut, expérience)
- Bouton d'action vers l'UI MLflow

---

## 9. Gestion des tags MLflow

### 9.1. Tags appliqués

| Tag (clé) | Valeurs possibles | Déclencheur |
|---|---|---|
| `deployment_status` | `go_prod` | Gate manuelle approuvée |
| `deployment_status` | `rejected` | MLScore < seuil OU gate manuelle rejetée |

### 9.2. Transitions de stage MLflow

| Situation | Stage MLflow |
|---|---|
| Modèle enregistré initialement | `None` |
| Modèle validé pour production | `Production` |
| Modèle rejeté | `Archived` |

### 9.3. Endpoints MLflow utilisés

| Action | Endpoint MLflow REST |
|---|---|
| Transition de stage | `POST /api/2.0/mlflow/model-versions/transition-stage` |
| Application de tag | `POST /api/2.0/mlflow/model-versions/set-tag` |

---

## 10. Variables d'environnement

### Variables existantes (inchangées)

| Variable | Valeur par défaut | Usage |
|---|---|---|
| `POSTGRES_USER` | `mlops` | Utilisateur PostgreSQL |
| `POSTGRES_PASSWORD` | `mlops` | Mot de passe PostgreSQL |
| `POSTGRES_DB` | `mlops` | Base de données principale |
| `DATABASE_URL` | `postgresql://mlops:mlops@db:5432/mlops` | Connexion API → DB |
| `API_URL` | `http://api:8000` | URL de l'API pour les workers |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URL MLflow |
| `POLL_INTERVAL` | `10` | (obsolète si RabbitMQ, conservé en fallback) |
| `OUTPUT_DIR` | `/app/outputs` | Répertoire des modèles entraînés |
| `MISTRAL_API_KEY` | — | Clé API Mistral (LLM juge RAGAS) |
| `MISTRAL_MODEL` | `mistral-small-latest` | Modèle Mistral pour RAGAS |
| `MLSCORE_THRESHOLD` | `0.7` | Seuil de validation du MLScore |

### Variables nouvelles

| Variable | Valeur par défaut | Usage |
|---|---|---|
| `RABBITMQ_URL` | `amqp://guest:guest@rabbitmq:5672/` | Connexion au broker RabbitMQ |
| `RABBITMQ_DEFAULT_USER` | `guest` | Utilisateur RabbitMQ |
| `RABBITMQ_DEFAULT_PASS` | `guest` | Mot de passe RabbitMQ |
| `RELEASE_SERVER_URL` | — | URL du serveur Digital.ai Release |
| `RELEASE_API_TOKEN` | — | Token d'authentification Release |
| `RELEASE_TEMPLATE_ID` | — | ID du template de release MLOps |
| `TEAMS_WEBHOOK_URL` | — | URL du webhook entrant Microsoft Teams |

---

## 11. Infrastructure Docker

### 11.1. Nouveaux services

```yaml
# ── RabbitMQ ───────────────────────────────────────────────────────
rabbitmq:
  image: rabbitmq:3-management
  environment:
    RABBITMQ_DEFAULT_USER: ${RABBITMQ_DEFAULT_USER:-guest}
    RABBITMQ_DEFAULT_PASS: ${RABBITMQ_DEFAULT_PASS:-guest}
  ports:
    - "5672:5672"    # AMQP
    - "15672:15672"  # UI Management
  volumes:
    - rabbitmq_data:/var/lib/rabbitmq
  healthcheck:
    test: ["CMD", "rabbitmq-diagnostics", "check_port_connectivity"]
    interval: 10s
    timeout: 5s
    retries: 5

# ── Release Worker ─────────────────────────────────────────────────
release-worker:
  build:
    context: .
    dockerfile: docker/Dockerfile.release-worker
  environment:
    API_URL: http://api:8000
    RABBITMQ_URL: amqp://guest:guest@rabbitmq:5672/
    RELEASE_SERVER_URL: ${RELEASE_SERVER_URL}
    RELEASE_API_TOKEN: ${RELEASE_API_TOKEN}
    RELEASE_TEMPLATE_ID: ${RELEASE_TEMPLATE_ID}
  depends_on:
    rabbitmq:
      condition: service_healthy
    api:
      condition: service_started
```

### 11.2. Services modifiés

| Service | Modification |
|---|---|
| `api` | Ajout de `pika` en dépendance, publication d'événements RabbitMQ |
| `training` | Migration polling → consommation RabbitMQ (fallback polling conservé) |
| `evaluation` | Migration polling → consommation RabbitMQ (fallback polling conservé) |

### 11.3. Nouveaux volumes

| Volume | Service | Usage |
|---|---|---|
| `rabbitmq_data` | `rabbitmq` | Persistance des messages et config RabbitMQ |

### 11.4. Cartographie complète des services

```
┌────────────────────────────────────────────────────────────────────────┐
│                          docker-compose                                │
│                                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │    UI    │  │   API    │  │  MLflow  │  │PostgreSQL│  │RabbitMQ│  │
│  │ Streamlit│  │ FastAPI  │  │ Tracking │  │          │  │        │  │
│  │  :8501   │  │  :8000   │  │  :5000   │  │  :5432   │  │  :5672 │  │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘  │ :15672 │  │
│       │              │                                     └───┬────┘  │
│       └──────►       ├──publish──────────────────────────────►│       │
│                      │                                         │       │
│              ┌───────┴───────┐                    ┌────────────┤       │
│              │               │                    │            │       │
│        ┌─────┴─────┐  ┌─────┴──────┐  ┌──────────┴──┐        │       │
│        │ Training  │  │ Evaluation │  │   Release   │◄───────┘       │
│        │  Worker   │  │   Worker   │  │   Worker    │                │
│        └───────────┘  └────────────┘  └─────────────┘                │
│                                              │                        │
│                                              ▼                        │
│                                    Digital.ai Release                  │
│                                    (serveur externe)                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Matrice des décisions

### Décisions de branchement dans le pipeline

| Point de décision | Condition | Branche | Action |
|---|---|---|---|
| Fin de l'évaluation | `MLScore ≥ MLSCORE_THRESHOLD` | Succès | Avance vers la gate manuelle |
| Fin de l'évaluation | `MLScore < MLSCORE_THRESHOLD` | Échec | Tag `rejected`, release avortée |
| Gate manuelle | Approbateur approuve | Succès | Tag `go_prod`, stage `Production` |
| Gate manuelle | Approbateur rejette | Échec | Tag `rejected`, stage `Archived` |

### Garanties d'idempotence

| Opération | Risque de doublon | Mitigation |
|---|---|---|
| Création de release | Un même `run_id` pourrait déclencher 2 releases | Vérifier l'existence d'une release pour ce `run_id` avant création |
| Tag MLflow | Un tag pourrait être appliqué deux fois | Les tags MLflow sont upsert par nature (pas de doublon) |
| Notification Teams | Un message pourrait être envoyé deux fois | Acceptable (préférer un doublon à un silence) |
| Transition de stage MLflow | Conflit si deux transitions concurrentes | Vérifier le stage actuel avant transition |

---

## 13. Comparaison Jira vs Digital.ai Release

### Pourquoi Digital.ai Release plutôt que Jira

| Critère | Jira | Digital.ai Release |
|---|---|---|
| **Rôle natif** | Tracker d'issues (passif) | Orchestrateur de releases (actif) |
| **Workflow** | Le système notifie Jira des changements | Release pilote et contrôle les étapes |
| **Gate manuelle** | Transition de ticket (pas d'approbation formelle) | Gate native avec rôles, SLA, audit trail |
| **Containers supplémentaires** | Jira Worker + Teams Worker (2 containers) | Release Worker seul (1 container) |
| **Webhook entrant** | Requis (`POST /webhook/jira` dans l'API) | Non requis (la gate gère nativement) |
| **Plugin** | Pas de plugin (uniquement API REST Jira) | Plugin Python avec tâches custom |
| **Traçabilité** | Historique du ticket | Pipeline visuel complet avec durées, logs, variables |
| **Scalabilité** | Limité aux transitions de ticket | Templates réutilisables, variables dynamiques, parallélisme |
| **Audit** | Changelog Jira | Audit trail complet des approbations et décisions |

### Ce qui resterait identique avec Jira

- RabbitMQ comme broker de messages
- Le calcul du MLScore et la logique de validation
- Les notifications Teams
- Le tagging MLflow

### Ce qui changerait avec Jira

- Un **Jira Worker** remplacerait le Release Worker (déplace les tickets via l'API Jira)
- Un endpoint **`POST /webhook/jira`** serait nécessaire dans l'API FastAPI pour recevoir les transitions de ticket
- La gate manuelle serait simulée par la transition "À tester → Terminé" dans le board Jira
- Pas de plugin Python côté Jira (logique entièrement dans le Jira Worker)
