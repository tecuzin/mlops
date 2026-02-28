from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="MLOps — Entraînement, Évaluation & Sécurité LLM", layout="wide")
st.title("MLOps — Entraînement, Évaluation & Sécurité LLM")


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


OWASP_LABELS = {
    "sec_prompt_injection": "LLM01 — Injection d'invites",
    "sec_output_handling": "LLM02 — Sorties non sécurisées",
    "sec_data_poisoning": "LLM03 — Empoisonnement des données",
    "sec_model_dos": "LLM04 — Déni de service",
    "sec_supply_chain": "LLM05 — Chaîne logistique",
    "sec_info_disclosure": "LLM06 — Divulgation d'informations",
    "sec_overreliance": "LLM09 — Dépendance excessive",
    "sec_model_theft": "LLM10 — Vol de modèle",
    "ml_sec_score": "MLSecScore global",
}

STATUS_ICONS = {
    "pending": "⏳",
    "training": "🏋️",
    "evaluating": "📊",
    "security_scanning": "🔒",
    "completed": "✅",
    "failed": "❌",
}

# ═══════════════════════════════════════════════════════════════════════
tab_config, tab_status, tab_results, tab_security = st.tabs([
    "⚙️ Configuration",
    "📡 Status",
    "📊 Résultats",
    "🔒 Sécurité",
])


# ── Onglet Configuration ──────────────────────────────────────────────
with tab_config:
    st.header("Nouvelle expérience")

    try:
        train_datasets = api_get("/datasets", dataset_type="train")
        eval_datasets = api_get("/datasets", dataset_type="eval")
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        train_datasets, eval_datasets = [], []

    if train_datasets or eval_datasets:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Modèle")
            experiment_name = st.text_input("Nom de l'expérience", value="mlops-default")
            model_name = st.text_input("Nom du run", placeholder="mistral-7b-rag-qa")
            model_id = st.text_input(
                "HuggingFace Model ID",
                placeholder="mistralai/Mistral-7B-v0.1",
            )
            task_type = st.selectbox("Type de tâche", ["finetune", "eval_only", "security_eval"])
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
            elif task_type == "security_eval":
                train_ds_options = {d["name"]: d["id"] for d in train_datasets}
                selected_train = st.selectbox(
                    "Dataset d'entraînement (optionnel, pour audit PII)",
                    options=["(aucun)"] + list(train_ds_options.keys()),
                )
                train_dataset_id = train_ds_options.get(selected_train)
            else:
                train_dataset_id = None

            if task_type == "security_eval":
                eval_dataset_id = None
                st.info("Pas de dataset d'évaluation requis pour l'analyse de sécurité.")
            else:
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
            elif task_type == "security_eval":
                st.info("Configuration de l'analyse de sécurité ci-contre.")
            else:
                st.info("Pas d'hyperparamètres pour l'évaluation seule.")

        with col_ragas:
            if task_type == "security_eval":
                st.subheader("Configuration sécurité (OWASP Top 10)")
                sec_modelscan = st.checkbox("ModelScan — analyse statique des artefacts", value=True)
                sec_data_audit = st.checkbox("Audit des données d'entraînement (PII)", value=True)
                sec_prompt_injection = st.checkbox("Injection d'invites (LLM01)", value=True)
                sec_pii_leakage = st.checkbox("Divulgation d'informations (LLM06)", value=True)
                sec_toxicity = st.checkbox("Toxicité / sorties non sécurisées (LLM02)", value=True)
                sec_bias = st.checkbox("Biais (discrimination)", value=True)
                sec_hallucination = st.checkbox("Hallucinations / sur-confiance (LLM09)", value=True)
                sec_dos = st.checkbox("Résilience DoS (LLM04)", value=True)
                with st.expander("Paramètres avancés"):
                    sec_max_probes = st.number_input("Probes max par catégorie", value=50, min_value=5, max_value=500, step=5)
                    sec_timeout = st.number_input("Timeout par probe (secondes)", value=300, min_value=30, max_value=3600, step=30)
            else:
                st.subheader("Métriques RAGAS")
                m_faithfulness = st.checkbox("Faithfulness", value=True)
                m_answer_relevancy = st.checkbox("Answer Relevancy", value=True)
                m_context_precision = st.checkbox("Context Precision", value=True)
                m_context_recall = st.checkbox("Context Recall", value=True)

        st.divider()

        if st.button("Valider & lancer le pipeline", type="primary"):
            if not model_name or not model_id:
                st.error("Le nom du run et le Model ID sont obligatoires.")
            elif task_type != "security_eval" and not eval_dataset_id:
                st.error("Un dataset d'évaluation est obligatoire.")
            else:
                payload: dict = {
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                    "model_id": model_id,
                    "task_type": task_type,
                    "train_dataset_id": train_dataset_id,
                    "eval_dataset_id": eval_dataset_id,
                    "register_model": register_model,
                }

                if task_type == "security_eval":
                    payload["security_config"] = {
                        "modelscan_enabled": sec_modelscan,
                        "training_data_audit": sec_data_audit,
                        "prompt_injection": sec_prompt_injection,
                        "pii_leakage": sec_pii_leakage,
                        "toxicity": sec_toxicity,
                        "bias": sec_bias,
                        "hallucination": sec_hallucination,
                        "dos_resilience": sec_dos,
                        "max_probes_per_category": sec_max_probes,
                        "timeout_per_probe_seconds": sec_timeout,
                    }
                else:
                    payload["ragas_metrics"] = {
                        "faithfulness": m_faithfulness,
                        "answer_relevancy": m_answer_relevancy,
                        "context_precision": m_context_precision,
                        "context_recall": m_context_recall,
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
        all_runs = []

    if not all_runs:
        st.info("Aucun run lancé pour le moment.")
    else:
        for run in all_runs:
            icon = STATUS_ICONS.get(run["status"], "❓")
            with st.expander(
                f"{icon} **{run['model_name']}** — {run['status'].upper()} — "
                f"ID {run['id']} — {run['created_at'][:19]}",
                expanded=run["status"] not in ("completed", "failed"),
            ):
                col_relaunch, col_spacer = st.columns([1, 4])
                with col_relaunch:
                    relaunch_clicked = st.button(
                        "🔄 Relancer",
                        key=f"relaunch_{run['id']}",
                        type="primary",
                    )
                if relaunch_clicked:
                    snapshot = run.get("config_snapshot", {})
                    payload = {
                        "experiment_name": snapshot.get("experiment_name", run["experiment_name"]),
                        "model_name": snapshot.get("model_name", run["model_name"]),
                        "model_id": snapshot.get("model_id", run["model_id"]),
                        "task_type": snapshot.get("task_type", run["task_type"]),
                        "train_dataset_id": snapshot.get("train_dataset_id"),
                        "eval_dataset_id": snapshot.get("eval_dataset_id"),
                        "register_model": snapshot.get("register_model", False),
                    }
                    if snapshot.get("ragas_metrics"):
                        payload["ragas_metrics"] = snapshot["ragas_metrics"]
                    if snapshot.get("security_config"):
                        payload["security_config"] = snapshot["security_config"]
                    if snapshot.get("training_params"):
                        payload["training_params"] = snapshot["training_params"]
                    try:
                        new_run = api_post("/runs", json=payload)
                        st.success(f"Run relancé ! Nouveau Run ID : **{new_run['id']}**")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur lors de la relance : {e}")

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
                    elif status == "security_scanning":
                        st.progress(0.50, text="Analyse de sécurité en cours...")
                    elif status == "completed":
                        st.progress(1.0, text="Terminé")
                    elif status == "failed":
                        st.progress(1.0, text="Échoué")

                    if run.get("results"):
                        st.markdown("**Résultats provisoires :**")
                        for r in run["results"]:
                            label = r["metric_name"]
                            if label.startswith("sec_"):
                                label = OWASP_LABELS.get(r["metric_name"], label.replace("sec_", "").replace("_", " ").title())
                            st.metric(label, f"{r['metric_value']:.4f}")

                if run.get("logs"):
                    with st.expander("Logs détaillés"):
                        st.code(run["logs"], language="text")


# ── Onglet Résultats ──────────────────────────────────────────────────
with tab_results:
    st.header("Résultats RAGAS & Comparaison")

    if st.button("Rafraîchir", key="refresh_results"):
        st.rerun()

    try:
        all_completed = [r for r in api_get("/runs") if r["status"] == "completed"]
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        all_completed = []

    ragas_runs = [r for r in all_completed if r["task_type"] != "security_eval"]

    if not ragas_runs:
        st.info("Aucun run d'entraînement / évaluation terminé pour le moment.")
    else:
        ragas_metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "ml_score"]
        training_metric_names = ["train_loss", "perplexity", "train_runtime", "train_samples_per_second"]

        try:
            all_datasets = {d["id"]: d["name"] for d in api_get("/datasets")}
        except Exception:
            all_datasets = {}

        def _lifecycle_tag(run):
            if run.get("mlflow_model_version"):
                return "finetuned"
            if run["task_type"] == "finetune":
                return "trained"
            return ""

        def _domain_tag(run):
            for key in ("train_dataset_id", "eval_dataset_id"):
                ds_id = run.get(key) or run.get("config_snapshot", {}).get(key)
                if ds_id:
                    name = all_datasets.get(ds_id, "").lower()
                    if "medical" in name:
                        return "medic"
                    if "legal" in name:
                        return "legal"
            return ""

        def _validation_tag(run):
            for r in run.get("results", []):
                if r["metric_name"] == "ml_score":
                    return "validated" if r["metric_value"] >= 0.7 else "rejected"
            return ""
        # ── Tableau des métriques RAGAS ───────────────────────
        st.subheader("Scores RAGAS")
        rows = []
        for run in ragas_runs:
            metrics = {r["metric_name"]: r["metric_value"] for r in run.get("results", [])}
            row = {
                "Run ID": run["id"],
                "Modèle": run["model_name"],
                "Model ID": run["model_id"],
                "Tâche": run["task_type"],
                "Lifecycle": _lifecycle_tag(run),
                "Domaine": _domain_tag(run),
                "Validation": _validation_tag(run),
                "Date": run["created_at"][:19],
            }
            for m in ragas_metric_names:
                if m in metrics:
                    row[m] = round(metrics[m], 4)
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        NON_METRIC_COLS = {"Run ID", "Modèle", "Model ID", "Tâche", "Lifecycle", "Domaine", "Validation", "Date"}
        metric_cols = [c for c in df.columns if c not in NON_METRIC_COLS]

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
            st.plotly_chart(fig)

            st.subheader("Radar des scores RAGAS")
            fig_radar = go.Figure()
            for _, row_data in df.iterrows():
                values = [row_data.get(m, 0) for m in metric_cols]
                values.append(values[0])
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
            st.plotly_chart(fig_radar)

        if len(ragas_runs) > 1 and metric_cols:
            st.subheader("Modèle champion")
            df_scores = df[metric_cols].copy()
            df["score_moyen"] = df_scores.mean(axis=1)
            champion_idx = df["score_moyen"].idxmax()
            champion = df.loc[champion_idx]
            st.success(
                f"Le meilleur modèle est **{champion['Modèle']}** "
                f"avec un score moyen de **{champion['score_moyen']:.4f}**"
            )

        # ── Tableau des métriques d'entraînement ──────────────
        st.divider()
        st.subheader("Métriques d'entraînement")
        train_rows = []
        for run in ragas_runs:
            metrics = {r["metric_name"]: r["metric_value"] for r in run.get("results", [])}
            row = {"Modèle": run["model_name"]}
            for m in training_metric_names:
                if m in metrics:
                    row[m] = round(metrics[m], 4)
            train_rows.append(row)

        df_train = pd.DataFrame(train_rows)
        st.dataframe(df_train, hide_index=True)

        # ── Export CSV ────────────────────────────────────────
        st.divider()
        st.subheader("Export")
        all_export_rows = []
        for run in ragas_runs:
            row = {
                "Run ID": run["id"],
                "Modèle": run["model_name"],
                "Model ID": run["model_id"],
                "Tâche": run["task_type"],
                "Date": run["created_at"][:19],
            }
            for r in run.get("results", []):
                row[r["metric_name"]] = r["metric_value"]
            all_export_rows.append(row)

        df_export = pd.DataFrame(all_export_rows)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger les résultats RAGAS (.csv)",
            data=csv_data,
            file_name="resultats_ragas.csv",
            mime="text/csv",
        )


# ── Onglet Sécurité ──────────────────────────────────────────────────
with tab_security:
    st.header("Évaluations de sécurité — OWASP Top 10 LLM")

    if st.button("Rafraîchir", key="refresh_security"):
        st.rerun()

    try:
        all_runs_sec = api_get("/runs")
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        all_runs_sec = []

    security_runs = [r for r in all_runs_sec if r["task_type"] == "security_eval"]
    completed_sec = [r for r in security_runs if r["status"] == "completed"]
    active_sec = [r for r in security_runs if r["status"] not in ("completed", "failed")]
    failed_sec = [r for r in security_runs if r["status"] == "failed"]

    if not security_runs:
        st.info(
            "Aucune évaluation de sécurité lancée pour le moment.\n\n"
            "Rendez-vous dans l'onglet **Configuration** pour créer un run de type `security_eval`."
        )
    else:
        # ── Active runs ───────────────────────────────────────
        if active_sec:
            st.subheader("Évaluations en cours")
            for run in active_sec:
                icon = STATUS_ICONS.get(run["status"], "🔒")
                st.info(f"{icon} **{run['model_name']}** — {run['status'].upper()} (ID {run['id']})")

        # ── Failed runs ───────────────────────────────────────
        if failed_sec:
            st.subheader("Évaluations échouées")
            for run in failed_sec:
                with st.expander(f"❌ {run['model_name']} (ID {run['id']})"):
                    if run.get("error_message"):
                        st.error(run["error_message"])
                    if run.get("logs"):
                        st.code(run["logs"][-2000:], language="text")

        # ── Completed results ─────────────────────────────────
        if completed_sec:
            st.subheader("Résultats de sécurité")

            owasp_metrics = [
                "sec_prompt_injection",
                "sec_output_handling",
                "sec_data_poisoning",
                "sec_model_dos",
                "sec_supply_chain",
                "sec_info_disclosure",
                "sec_overreliance",
                "sec_model_theft",
            ]

            sec_rows = []
            for run in completed_sec:
                metrics = {r["metric_name"]: r["metric_value"] for r in run.get("results", [])}
                row = {
                    "Run ID": run["id"],
                    "Modèle": run["model_name"],
                    "Date": run["created_at"][:19],
                }
                for m in owasp_metrics:
                    label = OWASP_LABELS.get(m, m)
                    row[label] = round(metrics.get(m, 0), 4)
                row["MLSecScore"] = round(metrics.get("ml_sec_score", 0), 4)
                sec_rows.append(row)

            df_sec = pd.DataFrame(sec_rows)
            st.dataframe(df_sec, hide_index=True)

            # ── MLSecScore badges ─────────────────────────────
            st.subheader("MLSecScore")
            badge_cols = st.columns(len(completed_sec))
            for i, run in enumerate(completed_sec):
                mlsecscore = next(
                    (r["metric_value"] for r in run.get("results", []) if r["metric_name"] == "ml_sec_score"),
                    None,
                )
                with badge_cols[i]:
                    if mlsecscore is not None:
                        color = "green" if mlsecscore >= 0.7 else ("orange" if mlsecscore >= 0.4 else "red")
                        st.metric(
                            label=run["model_name"],
                            value=f"{mlsecscore:.4f}",
                        )
                        st.markdown(f":{color}[{'Bon' if mlsecscore >= 0.7 else 'Moyen' if mlsecscore >= 0.4 else 'Faible'}]")

            # ── Radar chart ───────────────────────────────────
            owasp_radar_labels = [OWASP_LABELS[m] for m in owasp_metrics]

            st.subheader("Radar de sécurité OWASP Top 10")
            fig_sec = go.Figure()
            for run in completed_sec:
                metrics = {r["metric_name"]: r["metric_value"] for r in run.get("results", [])}
                values = [metrics.get(m, 0) for m in owasp_metrics]
                values.append(values[0])
                fig_sec.add_trace(go.Scatterpolar(
                    r=values,
                    theta=owasp_radar_labels + [owasp_radar_labels[0]],
                    name=run["model_name"],
                    fill="toself",
                    opacity=0.6,
                ))
            fig_sec.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Profil de sécurité OWASP Top 10",
                height=500,
            )
            st.plotly_chart(fig_sec)

            # ── Bar chart ─────────────────────────────────────
            st.subheader("Comparaison par catégorie OWASP")
            bar_rows = []
            for run in completed_sec:
                metrics = {r["metric_name"]: r["metric_value"] for r in run.get("results", [])}
                for m in owasp_metrics:
                    bar_rows.append({
                        "Modèle": run["model_name"],
                        "Catégorie": OWASP_LABELS.get(m, m),
                        "Score": metrics.get(m, 0),
                    })
            df_bar = pd.DataFrame(bar_rows)
            fig_bar = px.bar(
                df_bar,
                x="Catégorie",
                y="Score",
                color="Modèle",
                barmode="group",
                title="Scores de sécurité par catégorie OWASP",
                range_y=[0, 1],
            )
            fig_bar.update_layout(height=450, xaxis_tickangle=-30)
            st.plotly_chart(fig_bar)

            # ── Detailed logs per run ─────────────────────────
            st.divider()
            st.subheader("Détails par évaluation")
            for run in completed_sec:
                with st.expander(f"🔒 {run['model_name']} — ID {run['id']}"):
                    metrics = {r["metric_name"]: r["metric_value"] for r in run.get("results", [])}
                    cols = st.columns(4)
                    for idx, m in enumerate(owasp_metrics):
                        label = OWASP_LABELS.get(m, m)
                        val = metrics.get(m, 0)
                        with cols[idx % 4]:
                            st.metric(label.split(" — ")[0], f"{val:.4f}")
                    if run.get("logs"):
                        with st.expander("Logs"):
                            st.code(run["logs"][-3000:], language="text")

            # ── Export CSV ────────────────────────────────────
            st.divider()
            st.subheader("Export")
            csv_sec = df_sec.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Télécharger les résultats sécurité (.csv)",
                data=csv_sec,
                file_name="resultats_securite.csv",
                mime="text/csv",
            )
