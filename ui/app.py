from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="MLOps — Entraînement & Évaluation RAGAS", layout="wide")
st.title("MLOps — Entraînement & Évaluation RAGAS")


def api_get(path: str, **params):
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        resp = client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()


def api_post(path: str, **kwargs):
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        resp = client.post(path, **kwargs)
        resp.raise_for_status()
        return resp.json()


# ═══════════════════════════════════════════════════════════════════════
tab_config, tab_status, tab_results = st.tabs([
    "Configuration",
    "Status",
    "Résultats",
])


# ── Onglet Configuration ──────────────────────────────────────────────
with tab_config:
    st.header("Nouvelle expérience")

    try:
        train_datasets = api_get("/datasets", dataset_type="train")
        eval_datasets = api_get("/datasets", dataset_type="eval")
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Modèle")
        experiment_name = st.text_input("Nom de l'expérience", value="mlops-default")
        model_name = st.text_input("Nom du run", placeholder="mistral-7b-rag-qa")
        model_id = st.text_input(
            "HuggingFace Model ID",
            placeholder="mistralai/Mistral-7B-v0.1",
        )
        task_type = st.selectbox("Type de tâche", ["finetune", "eval_only"])
        register_model = st.checkbox("Enregistrer dans le Model Registry")

    with col2:
        st.subheader("Datasets")
        if task_type == "finetune":
            train_ds_options = {d["name"]: d["id"] for d in train_datasets}
            selected_train = st.selectbox(
                "Dataset d'entraînement",
                options=list(train_ds_options.keys()),
            )
            train_dataset_id = train_ds_options.get(selected_train)
        else:
            train_dataset_id = None

        eval_ds_options = {d["name"]: d["id"] for d in eval_datasets}
        selected_eval = st.selectbox(
            "Dataset d'évaluation",
            options=list(eval_ds_options.keys()),
        )
        eval_dataset_id = eval_ds_options.get(selected_eval)

    st.divider()

    col_hp, col_ragas = st.columns(2)

    with col_hp:
        st.subheader("Hyperparamètres")
        if task_type == "finetune":
            epochs = st.slider("Epochs", 1, 20, 3)
            batch_size = st.select_slider("Batch size", [1, 2, 4, 8, 16], value=4)
            learning_rate = st.number_input("Learning rate", value=2e-5, format="%.1e", step=1e-5)
            warmup_steps = st.number_input("Warmup steps", value=100, step=10)
            max_seq_length = st.select_slider("Max seq length", [128, 256, 512, 768, 1024, 2048], value=512)
            grad_accum = st.select_slider("Gradient accumulation steps", [1, 2, 4, 8, 16], value=4)
            fp16 = st.checkbox("FP16 (mixed precision)", value=True)

            with st.expander("Configuration LoRA (avancé)"):
                use_lora = st.checkbox("Activer LoRA", value=True)
                if use_lora:
                    lora_r = st.select_slider("LoRA rank (r)", [4, 8, 16, 32, 64], value=16)
                    lora_alpha = st.select_slider("LoRA alpha", [8, 16, 32, 64, 128], value=32)
                    lora_dropout = st.slider("LoRA dropout", 0.0, 0.5, 0.05, 0.01)
                    lora_modules = st.multiselect(
                        "Target modules",
                        ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        default=["q_proj", "v_proj"],
                    )
        else:
            st.info("Pas d'hyperparamètres pour l'évaluation seule.")

    with col_ragas:
        st.subheader("Métriques RAGAS")
        m_faithfulness = st.checkbox("Faithfulness", value=True)
        m_answer_relevancy = st.checkbox("Answer Relevancy", value=True)
        m_context_precision = st.checkbox("Context Precision", value=True)
        m_context_recall = st.checkbox("Context Recall", value=True)

    st.divider()

    if st.button("Valider & lancer le pipeline", type="primary", use_container_width=True):
        if not model_name or not model_id:
            st.error("Le nom du run et le Model ID sont obligatoires.")
        elif not eval_dataset_id:
            st.error("Un dataset d'évaluation est obligatoire.")
        else:
            payload = {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "model_id": model_id,
                "task_type": task_type,
                "train_dataset_id": train_dataset_id,
                "eval_dataset_id": eval_dataset_id,
                "ragas_metrics": {
                    "faithfulness": m_faithfulness,
                    "answer_relevancy": m_answer_relevancy,
                    "context_precision": m_context_precision,
                    "context_recall": m_context_recall,
                },
                "register_model": register_model,
            }
            if task_type == "finetune":
                lora_cfg = None
                if use_lora:
                    lora_cfg = {
                        "r": lora_r,
                        "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout,
                        "target_modules": lora_modules,
                    }
                payload["training_params"] = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "max_seq_length": max_seq_length,
                    "gradient_accumulation_steps": grad_accum,
                    "fp16": fp16,
                    "lora": lora_cfg,
                }

            try:
                result = api_post("/runs", json=payload)
                st.success(f"Pipeline lancé ! Run ID : **{result['id']}**")
            except Exception as e:
                st.error(f"Erreur lors du lancement : {e}")


# ── Onglet Status ─────────────────────────────────────────────────────
with tab_status:
    st.header("Suivi des pipelines")

    if st.button("Rafraîchir", key="refresh_status"):
        st.rerun()

    try:
        all_runs = api_get("/runs")
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        st.stop()

    if not all_runs:
        st.info("Aucun run lancé pour le moment.")
    else:
        STATUS_ICONS = {
            "pending": "⏳",
            "training": "🏋️",
            "evaluating": "📊",
            "completed": "✅",
            "failed": "❌",
        }

        for run in all_runs:
            icon = STATUS_ICONS.get(run["status"], "❓")
            with st.expander(
                f"{icon} **{run['model_name']}** — {run['status'].upper()} — "
                f"ID {run['id']} — {run['created_at'][:19]}",
                expanded=run["status"] not in ("completed", "failed"),
            ):
                col_info, col_progress = st.columns([1, 2])

                with col_info:
                    st.markdown(f"**Modèle :** `{run['model_id']}`")
                    st.markdown(f"**Tâche :** {run['task_type']}")
                    st.markdown(f"**Expérience :** {run['experiment_name']}")
                    if run.get("mlflow_run_id"):
                        st.markdown(f"**MLflow Run :** `{run['mlflow_run_id']}`")
                    if run.get("error_message"):
                        st.error(run["error_message"])

                with col_progress:
                    status = run["status"]
                    if status == "pending":
                        st.progress(0.0, text="En attente...")
                    elif status == "training":
                        st.progress(0.33, text="Entraînement en cours...")
                    elif status == "evaluating":
                        st.progress(0.66, text="Évaluation en cours...")
                    elif status == "completed":
                        st.progress(1.0, text="Terminé")
                    elif status == "failed":
                        st.progress(1.0, text="Échoué")

                    if run.get("results"):
                        st.markdown("**Résultats provisoires :**")
                        for r in run["results"]:
                            st.metric(r["metric_name"], f"{r['metric_value']:.4f}")

                if run.get("logs"):
                    with st.expander("Logs détaillés"):
                        st.code(run["logs"], language="text")


# ── Onglet Résultats ──────────────────────────────────────────────────
with tab_results:
    st.header("Résultats & Comparaison")

    try:
        completed_runs = [r for r in api_get("/runs") if r["status"] == "completed"]
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        st.stop()

    if not completed_runs:
        st.info("Aucun run terminé pour le moment.")
    else:
        rows = []
        for run in completed_runs:
            row = {
                "Run ID": run["id"],
                "Modèle": run["model_name"],
                "Model ID": run["model_id"],
                "Tâche": run["task_type"],
                "Date": run["created_at"][:19],
            }
            for r in run.get("results", []):
                row[r["metric_name"]] = r["metric_value"]
            rows.append(row)

        df = pd.DataFrame(rows)
        st.subheader("Tableau des scores")
        st.dataframe(df, use_container_width=True, hide_index=True)

        metric_cols = [c for c in df.columns if c not in ("Run ID", "Modèle", "Model ID", "Tâche", "Date")]

        if metric_cols:
            st.subheader("Comparaison graphique")

            df_melted = df.melt(
                id_vars=["Modèle"],
                value_vars=metric_cols,
                var_name="Métrique",
                value_name="Score",
            )
            fig = px.bar(
                df_melted,
                x="Métrique",
                y="Score",
                color="Modèle",
                barmode="group",
                title="Scores RAGAS par modèle",
                range_y=[0, 1],
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Radar des scores")
            fig_radar = go.Figure()
            for _, row_data in df.iterrows():
                values = [row_data.get(m, 0) for m in metric_cols]
                values.append(values[0])  # close the polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metric_cols + [metric_cols[0]],
                    name=row_data["Modèle"],
                    fill="toself",
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Radar des métriques RAGAS",
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Champion model
        if len(completed_runs) > 1 and metric_cols:
            st.subheader("Modèle champion")
            df["score_moyen"] = df[metric_cols].mean(axis=1)
            champion = df.loc[df["score_moyen"].idxmax()]
            st.success(
                f"Le meilleur modèle est **{champion['Modèle']}** "
                f"avec un score moyen de **{champion['score_moyen']:.4f}**"
            )

        # CSV export
        st.subheader("Export")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger les résultats (.csv)",
            data=csv_data,
            file_name="resultats_ragas.csv",
            mime="text/csv",
        )
