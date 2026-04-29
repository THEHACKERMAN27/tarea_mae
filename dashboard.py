"""
dashboard.py

Dashboard interactivo del pipeline CU Venta.

Ejecutar:
    streamlit run dashboard.py
"""

import glob
import json
import os

import pandas as pd
import streamlit as st

from config import MODEL_DIR, MONITORING_DIR, POST_DIR

st.set_page_config(
    page_title="CU Venta — Pipeline ML",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------

@st.cache_data
def load_tlv() -> pd.DataFrame | None:
    path = os.path.join(POST_DIR, "output_tlv.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_recall() -> pd.DataFrame | None:
    path = os.path.join(MONITORING_DIR, "recall_by_decile.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_metadata() -> dict | None:
    folders = sorted(glob.glob(os.path.join(MODEL_DIR, "*")), key=os.path.getmtime)
    if not folders:
        return None
    meta_files = glob.glob(os.path.join(folders[-1], "*.json"))
    if not meta_files:
        return None
    with open(meta_files[0]) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------

st.title("Dashboard — Pipeline CU Venta")
st.caption("Resultados del modelo EC OMNICANAL · Período de validación: 201912")

df      = load_tlv()
recall  = load_recall()
meta    = load_metadata()

if df is None:
    st.warning(
        "No se encontró el archivo de postprocesamiento. "
        "Ejecuta `python main.py` para generar los resultados."
    )
    st.stop()

st.divider()

# ------------------------------------------------------------------
# KPIs principales
# ------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Clientes scored",  f"{len(df):,}")
c2.metric("Score promedio",   f"{df['prob'].mean():.4f}")
c3.metric("TLV promedio",     f"{df['puntuacion_tlv'].mean():.6f}")
c4.metric("Monto promedio",   f"S/ {df['monto'].mean():,.0f}")

if meta:
    perf = meta.get("performance", {})
    c5.metric(
        "AUC Validación",
        f"{perf.get('auc_val', '—'):.4f}",
        delta=f"Decay {perf.get('decay_pct', 0):.1f}%",
        delta_color="inverse",
    )

st.divider()

# ------------------------------------------------------------------
# Distribución de grupos de ejecución
# ------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Distribución de Grupos de Ejecución")
    group_counts = (
        df["grupo_ejec_tlv"]
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("Grupo")
        .rename("Clientes")
    )
    st.bar_chart(group_counts)

with col_right:
    st.subheader("Score y Monto Promedio por Grupo")
    group_stats = (
        df.groupby(df["grupo_ejec_tlv"].astype(int))
        .agg(
            score_promedio=("prob",          "mean"),
            monto_promedio=("monto",         "mean"),
            tlv_promedio  =("puntuacion_tlv","mean"),
            clientes      =("prob",          "count"),
        )
        .reset_index()
        .rename(columns={"grupo_ejec_tlv": "grupo"})
        .sort_values("grupo")
    )
    st.dataframe(group_stats, width="stretch", hide_index=True)

st.divider()

# ------------------------------------------------------------------
# Recall acumulado por decil
# ------------------------------------------------------------------

if recall is not None:
    st.subheader("Recall Acumulado por Decil (score decreciente)")
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.line_chart(
            recall.set_index("decil")["recall_acumulado"],
            width="stretch",
        )
    with col_table:
        st.dataframe(recall, width="stretch", hide_index=True)

st.divider()

# ------------------------------------------------------------------
# Top-N clientes por puntuación TLV
# ------------------------------------------------------------------

st.subheader("Top Clientes por Puntuación TLV")

n_top = st.slider("Número de clientes", min_value=10, max_value=200, value=20, step=10)

display_cols = [
    c for c in ["key_value", "prob", "puntuacion_tlv", "grupo_ejec_tlv", "monto"]
    if c in df.columns
]
top_df = (
    df[display_cols]
    .sort_values("puntuacion_tlv", ascending=False)
    .head(n_top)
    .reset_index(drop=True)
)
top_df.index += 1
st.dataframe(top_df, width="stretch")

st.divider()

# ------------------------------------------------------------------
# Metadata del modelo
# ------------------------------------------------------------------

if meta:
    with st.expander("Detalle del modelo entrenado"):
        perf = meta.get("performance", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC Train", f"{perf.get('auc_train', '—'):.4f}")
        m2.metric("AUC Test",  f"{perf.get('auc_test',  '—'):.4f}")
        m3.metric("AUC Val",   f"{perf.get('auc_val',   '—'):.4f}")
        m4.metric("Decay %",   f"{perf.get('decay_pct', '—'):.2f}%")

        st.markdown("**Hiperparámetros**")
        st.json(meta.get("hyperparameters", {}))

        st.caption(f"Timestamp: {meta.get('timestamp', '—')}  ·  Run ID: {meta.get('run_id', '—')}")
