# MLOps — Entraînement & Évaluation RAGAS

Système MLOps conteneurisé pour le fine-tuning de LLMs et l'évaluation avec RAGAS, avec interface Streamlit, API FastAPI, et workers dédiés.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        docker-compose                            │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │    UI    │  │   API    │  │  MLflow  │  │   PostgreSQL   │  │
│  │ Streamlit│──│ FastAPI  │──│ Tracking │──│    (données)   │  │
│  │  :8501   │  │  :8000   │  │  :5000   │  │    :5432       │  │
│  └──────────┘  └─────┬────┘  └──────────┘  └────────────────┘  │
│                      │                                           │
│              ┌───────┴───────┐                                   │
│              │               │                                   │
│        ┌─────┴─────┐  ┌─────┴──────┐                            │
│        │ Training  │  │ Evaluation │                             │
│        │  Worker   │  │   Worker   │                             │
│        └───────────┘  └────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

### Conteneurs

| Conteneur | Rôle | Port | Image |
|---|---|---|---|
| **db** | Base de données PostgreSQL — stocke les runs, datasets, résultats, et les données MLflow | 5432 | `postgres:16-alpine` |
| **api** | API REST FastAPI — CRUD des runs/datasets, point d'entrée des workers et de l'UI | 8000 | Custom (`Dockerfile.api`) |
| **ui** | Interface Streamlit — configuration, suivi, résultats | 8501 | Custom (`Dockerfile.ui`) |
| **mlflow** | MLflow Tracking Server — tracking d'expériences, model registry | 5000 | `ghcr.io/mlflow/mlflow` |
| **training** | Worker d'entraînement — poll l'API, fine-tune les modèles via HuggingFace/LoRA | — | Custom (`Dockerfile.training`) |
| **evaluation** | Worker d'évaluation — poll l'API, évalue les modèles avec RAGAS | — | Custom (`Dockerfile.evaluation`) |

## Stack technique

| Composant | Outil |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| Base de données | PostgreSQL 16 |
| ORM | SQLAlchemy 2 |
| Tracking | MLflow |
| Évaluation | RAGAS |
| Fine-tuning | HuggingFace Transformers + PEFT/LoRA |
| Conteneurisation | Docker Compose |

## Structure du projet

```
mlops/
├── docker-compose.yml              # Orchestration des conteneurs
├── docker/
│   ├── Dockerfile.api              # Image API FastAPI
│   ├── Dockerfile.ui               # Image Streamlit
│   ├── Dockerfile.training         # Image worker entraînement
│   └── Dockerfile.evaluation       # Image worker évaluation
├── api/
│   ├── main.py                     # Endpoints REST (runs, datasets, résultats)
│   └── schemas.py                  # Schémas Pydantic entrée/sortie
├── ui/
│   └── app.py                      # Application Streamlit (3 onglets)
├── db/
│   ├── models.py                   # Modèles SQLAlchemy (PipelineRun, Dataset, RunResult)
│   ├── session.py                  # Connexion PostgreSQL
│   └── init_db.py                  # Création des tables + seed des datasets
├── workers/
│   ├── training/
│   │   └── worker.py               # Worker d'entraînement (poll + HuggingFace)
│   └── evaluation/
│       └── worker.py               # Worker d'évaluation (poll + RAGAS)
├── src/
│   ├── config.py                   # Modèles Pydantic pour config YAML
│   ├── training.py                 # Logique de fine-tuning (standalone)
│   ├── evaluation.py               # Logique d'évaluation RAGAS (standalone)
│   └── pipeline.py                 # Pipeline Prefect (standalone)
├── configs/                        # Configurations YAML d'exemple
│   ├── finetune_rag_qa.yaml
│   ├── eval_only.yaml
│   └── multi_dataset.yaml
├── data/
│   ├── train/                      # Jeux de données d'entraînement (JSONL)
│   │   ├── rag_qa_train.jsonl
│   │   ├── medical_qa_train.jsonl
│   │   └── legal_qa_train.jsonl
│   └── eval/                       # Jeux de données d'évaluation RAGAS (JSONL)
│       ├── ragas_eval.jsonl
│       ├── medical_ragas_eval.jsonl
│       └── legal_ragas_eval.jsonl
├── main.py                         # Point d'entrée CLI (mode standalone)
├── requirements.txt
├── .env.example
└── .dockerignore
```

## Démarrage rapide

### 1. Cloner et configurer

```bash
cp .env.example .env
# Éditer .env si besoin (HF_TOKEN pour les modèles privés)
```

### 2. Lancer tous les conteneurs

```bash
docker compose up --build
```

### 3. Accéder aux interfaces

| Interface | URL |
|---|---|
| **Streamlit UI** | http://localhost:8501 |
| **API FastAPI (docs)** | http://localhost:8000/docs |
| **MLflow UI** | http://localhost:5000 |

### 4. Utilisation

1. Ouvrir l'UI Streamlit sur http://localhost:8501
2. Onglet **Configuration** : choisir le modèle, le dataset, les hyperparamètres et les métriques RAGAS
3. Cliquer sur **Valider & lancer le pipeline**
4. Onglet **Status** : suivre la progression en temps réel
5. Onglet **Résultats** : consulter les scores, graphiques comparatifs et exporter en CSV

## API REST

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/datasets` | Liste des datasets (filtrable par type) |
| `POST` | `/runs` | Créer un nouveau run |
| `GET` | `/runs` | Liste des runs (filtrable par statut) |
| `GET` | `/runs/{id}` | Détail d'un run |
| `PATCH` | `/runs/{id}/status` | Mettre à jour le statut |
| `PATCH` | `/runs/{id}/logs` | Ajouter des logs |
| `POST` | `/runs/{id}/results` | Ajouter des résultats (métriques) |

## Base de données

### Tables

| Table | Description |
|---|---|
| `datasets` | Référence des jeux de données (nom, chemin, type, nombre de lignes) |
| `pipeline_runs` | Runs de pipeline (config, statut, logs, liens MLflow/Prefect) |
| `run_results` | Résultats métriques par run (nom, valeur) |

Au démarrage, le conteneur API initialise les tables et seed 6 datasets d'exemple.

## Format des données

### Entraînement (JSONL)

```json
{"question": "...", "answer": "...", "context": "..."}
```

### Évaluation RAGAS (JSONL)

```json
{"question": "...", "answer": "...", "contexts": ["...", "..."], "ground_truth": "..."}
```

## Flux de données

1. L'utilisateur configure un run via l'UI Streamlit
2. L'UI envoie la config à l'API (`POST /runs`)
3. L'API crée un `PipelineRun` en base (statut `pending`)
4. Le **training worker** poll l'API, détecte le run `pending` de type `finetune`, lance l'entraînement, met le statut à `training` puis `evaluating`
5. Le **evaluation worker** poll l'API, détecte les runs `evaluating` ou `eval_only` + `pending`, lance l'évaluation RAGAS, enregistre les scores et met le statut à `completed`
6. L'UI affiche les résultats en temps réel via polling de l'API

## Configuration YAML (mode standalone)

Les fichiers YAML dans `configs/` permettent aussi de lancer les pipelines en mode CLI :

```bash
python main.py configs/finetune_rag_qa.yaml
```

## Commandes utiles

```bash
# Lancer uniquement certains services
docker compose up db api ui

# Voir les logs d'un conteneur
docker compose logs -f training

# Reconstruire un conteneur
docker compose build training

# Arrêter et supprimer les volumes
docker compose down -v
```
