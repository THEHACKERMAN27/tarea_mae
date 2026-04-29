"""
api.py

API de inferencia REST para el modelo EC OMNICANAL.
Documentación interactiva disponible en /docs una vez levantada la API.

Ejecutar:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import glob
import json
import os
import pickle
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import MODEL_DIR, VARS_NUMERICAS

# ---------------------------------------------------------------------------
# Estado compartido del modelo
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def _find_latest_model_dir(base: str) -> str:
    dirs = sorted(
        [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)],
        key=os.path.getmtime,
    )
    if not dirs:
        raise RuntimeError(f"No se encontró ningún modelo en {base}")
    return dirs[-1]


def _load_model(model_dir: str):
    folder = _find_latest_model_dir(model_dir)

    pkl_files  = glob.glob(os.path.join(folder, "*.pkl"))
    json_files = glob.glob(os.path.join(folder, "*.json"))

    if not pkl_files or not json_files:
        raise RuntimeError(f"Artefactos incompletos en {folder}")

    with open(pkl_files[0], "rb") as f:
        model = pickle.load(f)

    with open(json_files[0], "r") as f:
        metadata = json.load(f)

    return model, metadata, folder


# ---------------------------------------------------------------------------
# Ciclo de vida de la aplicación
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    model, metadata, folder = _load_model(MODEL_DIR)
    _state["model"]    = model
    _state["metadata"] = metadata
    _state["folder"]   = folder
    print(f"Modelo cargado desde: {folder}")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EC OMNICANAL — API de Inferencia",
    description=(
        "Endpoint de predicción para el modelo de propensión CU Venta. "
        "Devuelve la probabilidad estimada de conversión por cliente."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class FeaturesInput(BaseModel):
    """
    Variables de entrada para una predicción individual.
    Todos los campos numéricos son opcionales; los valores ausentes
    se imputan internamente con el centinela -9999999.
    """
    nro_producto_6m:              float | None = Field(None, description="Número de productos últimos 6 meses")
    prom_uso_tc_rccsf3m:          float | None = None
    ctd_sms_received:             float | None = None
    max_usotcribksf06m:           float | None = None
    ctd_camptot06m:               float | None = None
    dsv_svppallsf06m:             float | None = None
    prm_svprmecs06m:              float | None = None
    ctd_app_productos_m1:         float | None = None
    ctd_campecsm01:               float | None = None
    lin_tcrrstsf03m:              float | None = None
    mnt_ptm:                      float | None = None
    dif_no_gestionado_4meses:     float | None = None
    max_campecs06m:               float | None = None
    beta_pctusotcr12m:            float | None = None
    rat_disefepnm01:              float | None = None
    flg_saltotppe12m:             float | None = None
    prom_sow_lintcribksf3m:       float | None = None
    openhtml_1m:                  float | None = None
    nprod_1m:                     float | None = None
    nro_transfer_6m:              float | None = None
    max_usotcrrstsf03m:           float | None = None
    prm_cnt_fee_amt_u7d:          float | None = None
    pas_avg6m_max12m:             float | None = None
    beta_saltotppe12m:            float | None = None
    seg_un:                       float | None = None
    ant_ultprdallsf:              float | None = None
    avg_sald_pas_3m:              float | None = None
    pas_1m_avg3m:                 float | None = None
    num_incrsaldispefe06m:        float | None = None
    cnl_age_p4m_p12m:             float | None = None
    cnl_atm_p4m_p12m:             float | None = None
    cre_lin_tc_rccibk_m07:        float | None = None
    prm_svprmlibdis06m:           float | None = None
    ingreso_neto:                 float | None = None
    max_nact_12m:                 float | None = None
    cre_sldtotfinprm03:           float | None = None
    dif_contacto_efectivo_10meses:float | None = None
    act_1m_avg3m:                 float | None = None
    monto_consumos_ecommerce_tc:  float | None = None
    ctd_camptotm01:               float | None = None
    prop_atm_4m:                  float | None = None
    prom_pct_saldopprcc6m:        float | None = None
    apppag_1m:                    float | None = None
    nro_configuracion_6m:         float | None = None
    act_avg6m_max12m:             float | None = None
    sldvig_tcrsrcf:               float | None = None
    prom_score_acepta_12meses:    float | None = None
    telefonos_6meses:             float | None = None
    pas_1m_avg6m:                 float | None = None
    ctd_camptototrcnl06m:         float | None = None
    prm_saltotrdpj03m:            float | None = None
    bpitrx_1m:                    float | None = None
    prm_lintcribksf03m:           float | None = None
    ctd_entrdm01:                 float | None = None
    avg_openhtml_6m:              float | None = None
    tea:                          float | None = None
    pct_usotcrm01:                float | None = None
    senthtml_1m:                  float | None = None
    ent_1erlntcrallsfm01_INTERBANK: float | None = None
    ent_1erlntcrallsfm01_OTRO:    float | None = None

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Probabilidad de conversión [0, 1]")
    model_name:  str   = Field(..., description="Algoritmo utilizado")
    model_ts:    str   = Field(..., description="Timestamp de entrenamiento del modelo")


class BatchInput(BaseModel):
    records: list[FeaturesInput] = Field(..., description="Lista de registros a puntuar")


class BatchResponse(BaseModel):
    n_records:     int
    probabilities: list[float]
    model_name:    str
    model_ts:      str


class HealthResponse(BaseModel):
    status:     str
    model_dir:  str
    model_name: str
    auc_val:    float | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dataframe(features: FeaturesInput) -> pd.DataFrame:
    row = features.model_dump()
    df  = pd.DataFrame([row])
    for col in df.columns:
        if df[col].dtype in ["float64", "float32"]:
            df[col] = df[col].fillna(-9999999).round(4)
    return df


def _predict(df: pd.DataFrame) -> np.ndarray:
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible.",
        )
    try:
        return model.predict_proba(df)[:, 1]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error en la inferencia: {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Verifica que la API esté operativa y que el modelo esté cargado."""
    meta = _state.get("metadata", {})
    return {
        "status":     "ok" if _state.get("model") else "degraded",
        "model_dir":  _state.get("folder", "—"),
        "model_name": meta.get("ml_name", "—"),
        "auc_val":    meta.get("performance", {}).get("auc_val"),
    }


@app.get("/model/info", tags=["Modelo"])
def model_info():
    """Devuelve la metadata completa del modelo en producción."""
    if not _state.get("metadata"):
        raise HTTPException(status_code=404, detail="Metadata no disponible.")
    return JSONResponse(content=_state["metadata"])


@app.post("/predict", response_model=PredictionResponse, tags=["Inferencia"])
def predict(payload: FeaturesInput):
    """
    Puntúa un único cliente.

    Devuelve la probabilidad de conversión estimada por el modelo.
    Los campos no enviados se imputan automáticamente con -9999999.
    """
    df  = _to_dataframe(payload)
    prob = float(_predict(df)[0])
    meta = _state["metadata"]
    return {
        "probability": round(prob, 6),
        "model_name":  meta.get("ml_name", "—"),
        "model_ts":    meta.get("timestamp", "—"),
    }


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inferencia"])
def predict_batch(payload: BatchInput):
    """
    Puntúa un lote de clientes en una sola llamada.

    Acepta hasta 10 000 registros por petición.
    """
    if len(payload.records) > 10_000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="El lote excede el límite de 10 000 registros.",
        )

    frames = [_to_dataframe(r) for r in payload.records]
    df     = pd.concat(frames, ignore_index=True)
    probs  = _predict(df).tolist()
    meta   = _state["metadata"]

    return {
        "n_records":     len(probs),
        "probabilities": [round(p, 6) for p in probs],
        "model_name":    meta.get("ml_name", "—"),
        "model_ts":      meta.get("timestamp", "—"),
    }
