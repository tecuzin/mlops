#!/usr/bin/env bash
# Script pour relancer l'entraînement sur 2 modèles + éval sécurité
# Usage: ./scripts/run_full_test.sh
# Prérequis: Docker Desktop démarré

set -e
cd "$(dirname "$0")/.."
API_URL="${API_URL:-http://localhost:8000}"

echo "=== 1. Vérification Docker ==="
if ! docker info &>/dev/null; then
  echo "ERREUR: Docker n'est pas démarré. Lancez Docker Desktop et réessayez."
  exit 1
fi

echo "=== 2. Démarrage de la stack ==="
docker compose up -d --build
echo "Attente du démarrage des services (60s)..."
sleep 60

echo "=== 3. Vérification de l'API ==="
for i in {1..12}; do
  if curl -sf "$API_URL/datasets" >/dev/null 2>&1; then
    echo "API disponible."
    break
  fi
  echo "Attente API... ($i/12)"
  sleep 5
done
if ! curl -sf "$API_URL/datasets" >/dev/null 2>&1; then
  echo "ERREUR: API non disponible après 60s."
  docker compose logs api --tail 30
  exit 1
fi

echo "=== 4. Récupération des IDs de datasets ==="
DATASETS=$(curl -sf "$API_URL/datasets")
TRAIN_ID=$(echo "$DATASETS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for x in d:
    if x.get('name')=='rag_qa_train':
        print(x['id'])
        break
else:
    print(1)
")
EVAL_ID=$(echo "$DATASETS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for x in d:
    if x.get('name')=='ragas_eval':
        print(x['id'])
        break
else:
    print(4)
")
echo "  train_dataset_id=$TRAIN_ID (rag_qa_train)"
echo "  eval_dataset_id=$EVAL_ID (ragas_eval)"

TRAINING_PARAMS='{"epochs":1,"batch_size":2,"learning_rate":2e-5,"warmup_steps":5,"max_seq_length":256,"gradient_accumulation_steps":2,"fp16":true,"lora":{"r":8,"lora_alpha":16,"lora_dropout":0.05,"target_modules":["q_proj","v_proj"]}}'

echo "=== 5. Création des runs (2 finetune + 2 security_eval) ==="

# Run 1: Mistral-7B finetune
echo "  → Run finetune: mistral-7b-rag-qa"
R1=$(curl -sf -X POST "$API_URL/runs" -H "Content-Type: application/json" -d "{
  \"experiment_name\": \"mlops-test\",
  \"model_name\": \"mistral-7b-rag-qa\",
  \"model_id\": \"mistralai/Mistral-7B-v0.1\",
  \"task_type\": \"finetune\",
  \"train_dataset_id\": $TRAIN_ID,
  \"eval_dataset_id\": $EVAL_ID,
  \"training_params\": $TRAINING_PARAMS,
  \"register_model\": false
}")
ID1=$(echo "$R1" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "    → run_id=$ID1"

# Run 2: Phi-2 finetune
echo "  → Run finetune: phi2-rag-qa"
R2=$(curl -sf -X POST "$API_URL/runs" -H "Content-Type: application/json" -d "{
  \"experiment_name\": \"mlops-test\",
  \"model_name\": \"phi2-rag-qa\",
  \"model_id\": \"microsoft/phi-2\",
  \"task_type\": \"finetune\",
  \"train_dataset_id\": $TRAIN_ID,
  \"eval_dataset_id\": $EVAL_ID,
  \"training_params\": $TRAINING_PARAMS,
  \"register_model\": false
}")
ID2=$(echo "$R2" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "    → run_id=$ID2"

# Run 3: Security eval Mistral (JSON complet en simple quotes pour éviter les problèmes de quoting)
echo "  → Run security_eval: mistral-7b-rag-qa"
R3=$(curl -sf -X POST "$API_URL/runs" -H "Content-Type: application/json" -d '{"experiment_name":"mlops-test","model_name":"mistral-7b-rag-qa","model_id":"mistralai/Mistral-7B-v0.1","task_type":"security_eval","train_dataset_id":'$TRAIN_ID',"security_config":{"modelscan_enabled":true,"training_data_audit":true,"prompt_injection":true,"pii_leakage":true,"toxicity":true,"bias":true,"hallucination":true,"dos_resilience":true,"max_probes_per_category":10,"timeout_per_probe_seconds":120}}')
ID3=$(echo "$R3" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "    → run_id=$ID3"

# Run 4: Security eval Phi-2
echo "  → Run security_eval: phi2-rag-qa"
R4=$(curl -sf -X POST "$API_URL/runs" -H "Content-Type: application/json" -d '{"experiment_name":"mlops-test","model_name":"phi2-rag-qa","model_id":"microsoft/phi-2","task_type":"security_eval","train_dataset_id":'$TRAIN_ID',"security_config":{"modelscan_enabled":true,"training_data_audit":true,"prompt_injection":true,"pii_leakage":true,"toxicity":true,"bias":true,"hallucination":true,"dos_resilience":true,"max_probes_per_category":10,"timeout_per_probe_seconds":120}}')
ID4=$(echo "$R4" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "    → run_id=$ID4"

echo ""
echo "=== 6. Runs créés — Suivi ==="
echo "  Finetune:     $ID1 (mistral-7b), $ID2 (phi2)"
echo "  Security:     $ID3 (mistral-7b), $ID4 (phi2)"
echo ""
echo "  UI Streamlit:  http://localhost:8501"
echo "  API docs:     http://localhost:8000/docs"
echo "  MLflow:       http://localhost:5001"
echo ""
echo "  Suivre les logs: docker compose logs -f training evaluation security"
