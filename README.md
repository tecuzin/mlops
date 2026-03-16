# MLOps — Entraînement, Évaluation, Sécurité et Lakehouse

Système MLOps conteneurisé pour entraîner, évaluer et auditer des modèles LLM, avec une entrée data lakehouse locale (medallion Bronze/Silver/Gold) pilotée par Spark.

## Architecture actuelle

Le projet tourne en local Docker avec 10 services :

- `db` (PostgreSQL)
- `api` (FastAPI)
- `ui` (Streamlit)
- `mlflow` (tracking/registry)
- `training`, `evaluation`, `security` (workers poll-based)
- `minio` (object storage lakehouse)
- `nessie` (catalog/versioning)
- `spark` (batch medallion)

## Composants principaux

| Composant | Rôle |
|---|---|
| FastAPI | Orchestration des runs (`/runs`, `/datasets`) |
| Workers | Exécution asynchrone training/evaluation/security |
| MLflow | Tracking, model registry, tags de cycle de vie |
| MinIO + Nessie + Spark | Ingestion et transformation lakehouse locale |
| Streamlit | Interface de pilotage et suivi des runs |

## Lakehouse medallion

Le flux data est structuré en trois couches :

1. **Bronze** — données brutes append-only + métadonnées d'ingestion
2. **Silver** — normalisation, typage et contrôles qualité
3. **Gold** — datasets entraînement-ready + métadonnées de snapshot

Le catalog `nessie` est utilisé activement pour publier les tables medallion (Bronze/Silver/Gold) via Spark avec configuration Iceberg/Nessie.  
Les workers consomment ensuite des exports figés validés par `catalog + table + reference + snapshot_id` avec fallback legacy si `LAKEHOUSE_ENABLED=false`.

Scripts utiles :

- `lakehouse/scripts/bootstrap_local.sh`
- `lakehouse/scripts/run_bronze.sh`
- `lakehouse/scripts/run_silver.sh`
- `lakehouse/scripts/run_gold.sh`
- `lakehouse/scripts/validate_reproducibility.sh`

## API runs et références lakehouse

`POST /runs` supporte désormais deux modes :

- **Legacy**: `train_dataset_id` / `eval_dataset_id`
- **Lakehouse**: `train_lakehouse_ref` / `eval_lakehouse_ref`

Exemple de référence lakehouse :

```json
{
  "catalog": "nessie",
  "namespace": "gold",
  "table": "rag_qa_train_ready",
  "reference": "main",
  "snapshot_id": "snapshot-20260316105535"
}
```

Les workers rejettent explicitement les incohérences `catalog/reference/snapshot_id` et résolvent ces références vers un dataset immuable.

## Démarrage rapide

```bash
cp .env.example .env
docker compose up --build
```

Interfaces :

- Streamlit: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- MLflow: `http://localhost:5001`
- MinIO console: `http://localhost:9001`
- Spark UI: `http://localhost:8081`

## Arborescence (résumé)

```text
mlops/
├── api/                        # FastAPI + schémas
├── workers/                    # training/evaluation/security + resolver lakehouse
├── lakehouse/
│   ├── jobs/                   # bronze/silver/gold spark jobs
│   ├── contracts/              # contrats de schéma/source mapping
│   ├── governance/             # guardrails medallion
│   ├── scripts/                # bootstrap/exécution/maintenance
│   ├── RUNBOOK.md
│   └── ROLLOUT_CHECKLIST.md
├── db/                         # modèles SQLAlchemy + seed/migrations
├── ui/                         # Streamlit
├── docker/                     # Dockerfiles services
└── openspec/changes/...        # artefacts OpenSpec de changement
```

## Schémas UML

Les schémas UML (composants, séquence, activités, classes) sont disponibles dans `docs/uml_schemas.md`.

## Commandes utiles

```bash
docker compose up --build
docker compose up -d db api minio nessie spark
docker compose logs -f training
docker compose down -v
```

## Notes importantes

- Les artefacts runtime (`captures/`, `lakehouse/warehouse/`, `lakehouse/metadata/`, `__pycache__/`) sont ignorés par Git.
- Le projet conserve le mode standalone (`main.py` + `src/`) en complément du mode Docker principal.
- Pour revenir au mode historique dataset-id/path, positionner `LAKEHOUSE_ENABLED=false` dans l'environnement des workers.
